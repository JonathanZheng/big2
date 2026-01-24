"""Training loop for Big 2 Deep Monte Carlo."""

import os
import copy
import random
import time
import multiprocessing as mp
from typing import List, Tuple, Optional
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from ..env import Big2Game, encode_state, encode_action, get_legal_moves, encode_move_history, encode_perfect_state, PERFECT_STATE_DIM
from ..models import SimpleNetwork, LSTMNetwork, CriticNetwork
from ..config import TRAINING_CONFIG, NETWORK_CONFIG, CRITIC_CONFIG, compute_epsilon, get_opponent_mix
from .buffer import ReplayBuffer
from .league import League
from ..agents import select_action_greedy_bot

# Global variable for checkpoint opponent model state dict (loaded once by main process)
_checkpoint_opponent_state_dict = None

# Global variable for league (initialized in train())
_league: Optional[League] = None


def select_action(
    game: Big2Game,
    player: int,
    model,
    epsilon: float,
    device: str
) -> Tuple[int, List]:
    """
    Select an action using epsilon-greedy policy (Stage 2).

    Supports both SimpleNetwork and LSTMNetwork.

    Args:
        game: Current game state
        player: Player index
        model: Policy network (SimpleNetwork or LSTMNetwork)
        epsilon: Exploration rate
        device: Device to run model on

    Returns:
        (action_index, legal_moves)
    """
    legal_moves = get_legal_moves(game, player)

    if len(legal_moves) == 0:
        raise ValueError(f"No legal moves for player {player}")

    # Epsilon-greedy
    if random.random() < epsilon:
        return random.randrange(len(legal_moves)), legal_moves

    # Encode state
    state = encode_state(game, player)
    state_tensor = torch.from_numpy(state).unsqueeze(0).to(device)

    # Check if model uses LSTM
    is_lstm = isinstance(model, LSTMNetwork)

    if is_lstm:
        # Encode move history for LSTM
        move_history = encode_move_history(game, max_moves=16)
        history_tensor = torch.from_numpy(move_history).unsqueeze(0).to(device)

    # Evaluate all legal actions
    q_values = []
    with torch.no_grad():
        for move in legal_moves:
            action = encode_action(move)
            action_tensor = torch.from_numpy(action).unsqueeze(0).to(device)

            if is_lstm:
                # LSTM forward pass
                q = model(history_tensor, state_tensor, action_tensor)
            else:
                # SimpleNetwork forward pass
                x = torch.cat([state_tensor, action_tensor], dim=1)
                q = model(x)

            q_values.append(q.item())

    # Select action with highest Q-value
    best_action = int(np.argmax(q_values))
    return best_action, legal_moves


def play_episode(
    model,
    epsilon: float,
    device: str,
    collect_perfect: bool = False
) -> Tuple[List[List[Tuple]], List[float]]:
    """
    Play one episode of self-play (Stage 2 with optional PTIE support).

    Args:
        model: Policy network (SimpleNetwork or LSTMNetwork)
        epsilon: Exploration rate
        device: Device to run model on
        collect_perfect: Whether to collect perfect states for PTIE

    Returns:
        (trajectories, rewards) where:
        - trajectories: List of 4 player trajectories, each containing
          (state, action, move_history) tuples, or
          (state, action, move_history, perfect_state) if collect_perfect=True
        - rewards: List of 4 final rewards
    """
    use_margin = TRAINING_CONFIG.get("use_margin_rewards", True)
    game = Big2Game(use_margin_rewards=use_margin)
    trajectories = [[], [], [], []]

    while not game.done:
        player = game.current_player

        # Encode state
        state = encode_state(game, player)

        # Encode move history
        move_history = encode_move_history(game, max_moves=16)

        # Encode perfect state for PTIE (if enabled)
        perfect_state = encode_perfect_state(game) if collect_perfect else None

        # Select action
        action_idx, legal_moves = select_action(game, player, model, epsilon, device)
        move = legal_moves[action_idx]

        # Encode action
        action_enc = encode_action(move)

        # Store in trajectory WITH move history (and optional perfect state)
        if collect_perfect:
            trajectories[player].append((
                state.copy(),
                action_enc.copy(),
                move_history.copy(),
                perfect_state.copy()
            ))
        else:
            trajectories[player].append((
                state.copy(),
                action_enc.copy(),
                move_history.copy()
            ))

        # Step game
        _, _, done, info = game.step(move)

    # Get final rewards
    rewards = info["all_rewards"]

    return trajectories, rewards


def soft_update(target_net, source_net, tau: float):
    """
    Soft update of target network parameters.

    θ_target = τ * θ_source + (1 - τ) * θ_target

    Args:
        target_net: Target network
        source_net: Source network
        tau: Soft update parameter
    """
    for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_(tau * source_param.data + (1 - tau) * target_param.data)


def evaluate_vs_random(
    model,
    num_games: int,
    device: str
) -> float:
    """
    Evaluate model against random opponents (Stage 2).

    Args:
        model: Policy network (SimpleNetwork or LSTMNetwork)
        num_games: Number of games to play
        device: Device to run model on

    Returns:
        Win rate (0.0 to 1.0)
    """
    wins = 0

    for _ in range(num_games):
        game = Big2Game()

        while not game.done:
            player = game.current_player

            if player == 0:
                # Model plays as player 0
                action_idx, legal_moves = select_action(game, player, model, epsilon=0.0, device=device)
            else:
                # Random opponents
                legal_moves = get_legal_moves(game, player)
                action_idx = random.randrange(len(legal_moves))

            move = legal_moves[action_idx]
            _, _, done, info = game.step(move)

        if info["winner"] == 0:
            wins += 1

    return wins / num_games


def evaluate_vs_greedy_bot(
    model,
    num_games: int,
    device: str
) -> float:
    """
    Evaluate model against greedy bot opponents (Stage 2).

    Args:
        model: Policy network (SimpleNetwork or LSTMNetwork)
        num_games: Number of games to play
        device: Device to run model on

    Returns:
        Win rate (0.0 to 1.0)
    """
    wins = 0
    for _ in range(num_games):
        game = Big2Game()

        while not game.done:
            player = game.current_player

            if player == 0:
                # Model plays as player 0
                action_idx, legal_moves = select_action(game, player, model, epsilon=0.0, device=device)
                move = legal_moves[action_idx]
            else:
                # Greedy bot opponents
                move = select_action_greedy_bot(game, player)

            _, _, done, info = game.step(move)

        if info["winner"] == 0:
            wins += 1

    return wins / num_games


def worker_play_episode(args):
    """
    Worker function to play one episode (Stage 2 with PTIE support).

    This function is designed to be called by multiprocessing workers.

    Args:
        args: Tuple of (model_state_dict, epsilon, seed, collect_perfect)
              or (model_state_dict, epsilon, seed) for backwards compatibility

    Returns:
        (trajectories, rewards) where:
        - trajectories: List of 4 player trajectories, each containing
          (state, action, move_history) or (state, action, move_history, perfect_state) tuples
        - rewards: List of 4 final rewards
    """
    # Handle both old and new argument formats
    if len(args) == 4:
        model_state_dict, epsilon, seed, collect_perfect = args
    else:
        model_state_dict, epsilon, seed = args
        collect_perfect = TRAINING_CONFIG.get("use_ptie", False)

    # Set random seed for this worker
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    # Create model on CPU (workers use CPU only)
    # Choose network type based on config
    if NETWORK_CONFIG.get("use_lstm", False):
        model = LSTMNetwork(**NETWORK_CONFIG)
    else:
        model = SimpleNetwork(**NETWORK_CONFIG)

    model.load_state_dict(model_state_dict)
    model.eval()
    device = "cpu"

    # Play episode (use margin rewards from config)
    use_margin = TRAINING_CONFIG.get("use_margin_rewards", True)
    game = Big2Game(use_margin_rewards=use_margin)
    trajectories = [[], [], [], []]

    with torch.no_grad():
        while not game.done:
            player = game.current_player

            # Encode state
            state = encode_state(game, player)

            # Encode move history
            move_history = encode_move_history(game, max_moves=16)

            # Encode perfect state for PTIE (if enabled)
            perfect_state = encode_perfect_state(game) if collect_perfect else None

            # Select action
            action_idx, legal_moves = select_action(game, player, model, epsilon, device)
            move = legal_moves[action_idx]

            # Encode action
            action_enc = encode_action(move)

            # Store in trajectory WITH move history (and optional perfect state)
            if collect_perfect:
                trajectories[player].append((
                    state.copy(),
                    action_enc.copy(),
                    move_history.copy(),
                    perfect_state.copy()
                ))
            else:
                trajectories[player].append((
                    state.copy(),
                    action_enc.copy(),
                    move_history.copy()
                ))

            # Step game
            _, _, done, info = game.step(move)

    # Get final rewards
    rewards = info["all_rewards"]

    return trajectories, rewards


def worker_play_episode_vs_greedy(args):
    """
    Worker function to play one episode with model as player 0 vs greedy opponents.

    Only collects trajectory for player 0 (the model).

    Args:
        args: Tuple of (model_state_dict, epsilon, seed, collect_perfect)
              or (model_state_dict, epsilon, seed) for backwards compatibility

    Returns:
        (trajectories, rewards) where:
        - trajectories: List of 4 player trajectories (only player 0 has data)
        - rewards: List of 4 final rewards
    """
    # Handle both old and new argument formats
    if len(args) == 4:
        model_state_dict, epsilon, seed, collect_perfect = args
    else:
        model_state_dict, epsilon, seed = args
        collect_perfect = TRAINING_CONFIG.get("use_ptie", False)

    # Set random seed for this worker
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    # Create model on CPU (workers use CPU only)
    if NETWORK_CONFIG.get("use_lstm", False):
        model = LSTMNetwork(**NETWORK_CONFIG)
    else:
        model = SimpleNetwork(**NETWORK_CONFIG)

    model.load_state_dict(model_state_dict)
    model.eval()
    device = "cpu"

    # Play episode (use margin rewards from config)
    use_margin = TRAINING_CONFIG.get("use_margin_rewards", True)
    game = Big2Game(use_margin_rewards=use_margin)
    trajectories = [[], [], [], []]  # Only player 0 will be populated

    with torch.no_grad():
        while not game.done:
            player = game.current_player

            if player == 0:
                # Model plays as player 0
                state = encode_state(game, player)
                move_history = encode_move_history(game, max_moves=16)
                perfect_state = encode_perfect_state(game) if collect_perfect else None
                action_idx, legal_moves = select_action(game, player, model, epsilon, device)
                move = legal_moves[action_idx]
                action_enc = encode_action(move)

                # Store in trajectory
                if collect_perfect:
                    trajectories[player].append((
                        state.copy(),
                        action_enc.copy(),
                        move_history.copy(),
                        perfect_state.copy()
                    ))
                else:
                    trajectories[player].append((
                        state.copy(),
                        action_enc.copy(),
                        move_history.copy()
                    ))
            else:
                # Greedy bot opponents
                move = select_action_greedy_bot(game, player)

            # Step game
            _, _, done, info = game.step(move)

    # Get final rewards
    rewards = info["all_rewards"]

    return trajectories, rewards


def worker_play_episode_vs_random(args):
    """
    Worker function to play one episode with model as player 0 vs random opponents.

    Only collects trajectory for player 0 (the model).

    Args:
        args: Tuple of (model_state_dict, epsilon, seed, collect_perfect)
              or (model_state_dict, epsilon, seed) for backwards compatibility

    Returns:
        (trajectories, rewards) where:
        - trajectories: List of 4 player trajectories (only player 0 has data)
        - rewards: List of 4 final rewards
    """
    # Handle both old and new argument formats
    if len(args) == 4:
        model_state_dict, epsilon, seed, collect_perfect = args
    else:
        model_state_dict, epsilon, seed = args
        collect_perfect = TRAINING_CONFIG.get("use_ptie", False)

    # Set random seed for this worker
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    # Create model on CPU (workers use CPU only)
    if NETWORK_CONFIG.get("use_lstm", False):
        model = LSTMNetwork(**NETWORK_CONFIG)
    else:
        model = SimpleNetwork(**NETWORK_CONFIG)

    model.load_state_dict(model_state_dict)
    model.eval()
    device = "cpu"

    # Play episode (use margin rewards from config)
    use_margin = TRAINING_CONFIG.get("use_margin_rewards", True)
    game = Big2Game(use_margin_rewards=use_margin)
    trajectories = [[], [], [], []]  # Only player 0 will be populated

    with torch.no_grad():
        while not game.done:
            player = game.current_player

            if player == 0:
                # Model plays as player 0
                state = encode_state(game, player)
                move_history = encode_move_history(game, max_moves=16)
                perfect_state = encode_perfect_state(game) if collect_perfect else None
                action_idx, legal_moves = select_action(game, player, model, epsilon, device)
                move = legal_moves[action_idx]
                action_enc = encode_action(move)

                # Store in trajectory
                if collect_perfect:
                    trajectories[player].append((
                        state.copy(),
                        action_enc.copy(),
                        move_history.copy(),
                        perfect_state.copy()
                    ))
                else:
                    trajectories[player].append((
                        state.copy(),
                        action_enc.copy(),
                        move_history.copy()
                    ))
            else:
                # Random opponents
                legal_moves = get_legal_moves(game, player)
                action_idx = random.randrange(len(legal_moves))
                move = legal_moves[action_idx]

            # Step game
            _, _, done, info = game.step(move)

    # Get final rewards
    rewards = info["all_rewards"]

    return trajectories, rewards


def worker_play_episode_vs_checkpoint(args):
    """
    Worker function to play one episode with model as player 0 vs checkpoint opponent.

    Training model as player 0 vs frozen checkpoint model as players 1-3.
    Checkpoint model uses greedy action selection (no exploration).

    Args:
        args: Tuple of (model_state_dict, checkpoint_state_dict, epsilon, seed, collect_perfect)
              For backwards compatibility, also accepts 4 args without collect_perfect.

    Returns:
        (trajectories, rewards) where:
        - trajectories: List of 4 player trajectories (only player 0 has data)
        - rewards: List of 4 final rewards
    """
    # Handle both old (4 args) and new (5 args with PTIE) formats
    if len(args) == 5:
        model_state_dict, checkpoint_state_dict, epsilon, seed, collect_perfect = args
    else:
        model_state_dict, checkpoint_state_dict, epsilon, seed = args
        collect_perfect = False

    # Set random seed for this worker
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    # Create training model on CPU
    if NETWORK_CONFIG.get("use_lstm", False):
        model = LSTMNetwork(**NETWORK_CONFIG)
        checkpoint_model = LSTMNetwork(**NETWORK_CONFIG)
    else:
        model = SimpleNetwork(**NETWORK_CONFIG)
        checkpoint_model = SimpleNetwork(**NETWORK_CONFIG)

    model.load_state_dict(model_state_dict)
    model.eval()

    # Load checkpoint opponent model
    checkpoint_model.load_state_dict(checkpoint_state_dict)
    checkpoint_model.eval()

    device = "cpu"

    # Play episode (use margin rewards from config)
    use_margin = TRAINING_CONFIG.get("use_margin_rewards", True)
    game = Big2Game(use_margin_rewards=use_margin)
    trajectories = [[], [], [], []]  # Only player 0 will be populated

    with torch.no_grad():
        while not game.done:
            player = game.current_player

            if player == 0:
                # Training model plays as player 0 with exploration
                state = encode_state(game, player)
                move_history = encode_move_history(game, max_moves=16)

                # Collect perfect state for PTIE if requested
                perfect_state = None
                if collect_perfect:
                    perfect_state = encode_perfect_state(game)

                action_idx, legal_moves = select_action(game, player, model, epsilon, device)
                move = legal_moves[action_idx]
                action_enc = encode_action(move)

                # Store in trajectory (include perfect_state if collecting)
                if collect_perfect:
                    trajectories[player].append((
                        state.copy(),
                        action_enc.copy(),
                        move_history.copy(),
                        perfect_state.copy()
                    ))
                else:
                    trajectories[player].append((
                        state.copy(),
                        action_enc.copy(),
                        move_history.copy()
                    ))
            else:
                # Checkpoint model opponents - greedy (no exploration)
                action_idx, legal_moves = select_action(game, player, checkpoint_model, 0.0, device)
                move = legal_moves[action_idx]

            # Step game
            _, _, done, info = game.step(move)

    # Get final rewards
    rewards = info["all_rewards"]

    return trajectories, rewards


def train(
    num_episodes: Optional[int] = None,
    checkpoint_path: Optional[str] = None,
    resume: bool = False,
    num_workers: int = 6,
    checkpoint_opponent_path: Optional[str] = None,
    use_league: bool = False,
    league_dir: Optional[str] = None
):
    """
    Main training loop with parallel workers.

    Args:
        num_episodes: Number of episodes to train (overrides config)
        checkpoint_path: Path to save checkpoints
        resume: Whether to resume from checkpoint
        num_workers: Number of parallel workers (default: 6)
        checkpoint_opponent_path: Path to frozen opponent model for curriculum learning
        use_league: Enable league training with opponent pool
        league_dir: Directory for league checkpoints
    """
    global _checkpoint_opponent_state_dict, _league

    # Get config
    config = TRAINING_CONFIG.copy()  # Make a copy to avoid modifying global
    if num_episodes is not None:
        config["num_episodes"] = num_episodes

    total_episodes = config["num_episodes"]

    # Setup device
    if torch.cuda.is_available():
        device = "cuda"
        print("Using CUDA (NVIDIA GPU) for acceleration")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("Using MPS (Metal Performance Shaders) for acceleration")
    else:
        device = "cpu"
        print("Using CPU")

    # Create model (LSTM or SimpleNetwork based on config)
    if NETWORK_CONFIG.get("use_lstm", False):
        model = LSTMNetwork(**NETWORK_CONFIG).to(device)
        target_model = copy.deepcopy(model)
        print("Using LSTM Network")
    else:
        model = SimpleNetwork(**NETWORK_CONFIG).to(device)
        target_model = copy.deepcopy(model)
        print("Using Simple Network")

    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    # PTIE: Create critic network if enabled
    use_ptie = config.get("use_ptie", False)
    critic = None
    critic_optimizer = None
    if use_ptie:
        critic = CriticNetwork(**CRITIC_CONFIG).to(device)
        critic_optimizer = optim.Adam(
            critic.parameters(),
            lr=config.get("critic_learning_rate", 1e-4)
        )
        print("Using PTIE with Critic Network")

    # Create replay buffer with normalization setting
    buffer = ReplayBuffer(
        capacity=config["buffer_size"],
        normalize_returns=config.get("normalize_returns", True)
    )

    # Initialize training state
    start_episode = 0
    best_win_rate = 0.0

    # Top-K checkpoint tracking
    top_k = config.get("top_k_checkpoints", 5)
    top_checkpoints = []  # List of (win_rate, episode, path)

    # Resume from checkpoint if requested
    if resume and checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        target_model.load_state_dict(checkpoint["target_model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_episode = checkpoint["episode"]
        best_win_rate = checkpoint.get("best_win_rate", 0.0)
        # Load critic if PTIE is enabled and critic was saved
        if use_ptie and critic is not None and "critic_state_dict" in checkpoint:
            critic.load_state_dict(checkpoint["critic_state_dict"])
            if "critic_optimizer_state_dict" in checkpoint:
                critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
            print("  Loaded critic from checkpoint")
        # Note: epsilon is computed fresh from episode/total_episodes

    # Compute initial epsilon based on start episode
    epsilon = compute_epsilon(
        start_episode, total_episodes,
        epsilon_start=config["epsilon_start"],
        epsilon_end=config["epsilon_end"],
        warmup_episodes=config.get("epsilon_warmup_episodes", 1000),
        schedule=config.get("epsilon_schedule", "cosine")
    )

    # Load checkpoint opponent model for curriculum learning
    checkpoint_opponent_path = checkpoint_opponent_path or config.get("checkpoint_opponent_path")
    _checkpoint_opponent_state_dict = None
    has_checkpoint_opponent = False

    if checkpoint_opponent_path and os.path.exists(checkpoint_opponent_path):
        print(f"Loading checkpoint opponent from: {checkpoint_opponent_path}")
        checkpoint_opponent = torch.load(checkpoint_opponent_path, map_location="cpu")
        _checkpoint_opponent_state_dict = checkpoint_opponent["model_state_dict"]
        has_checkpoint_opponent = True
    elif config.get("use_curriculum", False) and not use_league:
        print("Warning: use_curriculum=True but no checkpoint_opponent_path provided")
        print("  Checkpoint opponent games will use random opponents instead")

    # Initialize league training if enabled
    use_league = use_league or config.get("use_league", False)
    league_dir = league_dir or config.get("league_dir", "checkpoints/league")
    _league = None
    current_win_rate = 0.25  # Track current agent's win rate for skill-matched sampling

    if use_league:
        _league = League(league_dir, max_opponents=config.get("league_max_opponents", 20))
        loaded = _league.load_from_disk()
        if loaded:
            print(f"Loaded league from: {league_dir} ({len(_league)} opponents)")
        else:
            print(f"Initialized new league at: {league_dir}")

        # Take initial snapshot if configured and pool is empty
        if config.get("league_initial_snapshot", True) and len(_league) == 0:
            cpu_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
            _league.add_snapshot(cpu_state_dict, start_episode, current_win_rate)
            print(f"  Added initial snapshot to league pool")

    # Create checkpoint directory
    if checkpoint_path:
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)
    print(f"Episodes: {total_episodes} (starting from {start_episode})")
    print(f"Workers: {num_workers}")
    print(f"Buffer size: {config['buffer_size']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Learning rate: {config['learning_rate']}")
    print(f"Epsilon schedule: {config.get('epsilon_schedule', 'cosine')}")
    print(f"Epsilon: {epsilon:.3f} → {config['epsilon_end']:.3f}")
    print(f"Warmup episodes: {config.get('epsilon_warmup_episodes', 1000)}")
    print(f"Margin rewards: {config.get('use_margin_rewards', True)}")
    print(f"Return normalization: {config.get('normalize_returns', True)}")
    print(f"PTIE (Perfect Info Critic): {'ON' if use_ptie else 'OFF'}")
    if use_league:
        print(f"League training: ON")
        print(f"  League dir: {league_dir}")
        print(f"  Opponents in pool: {len(_league)}")
        print(f"  Snapshot frequency: {config.get('league_snapshot_freq', 2000)} episodes")
    if config.get("use_curriculum", False):
        opponent_mix = get_opponent_mix(start_episode, total_episodes, config)
        print(f"Curriculum learning: ON")
        print(f"  Initial mix: {opponent_mix['self_play']*100:.0f}% self-play, "
              f"{opponent_mix['greedy']*100:.0f}% greedy, "
              f"{opponent_mix['checkpoint']*100:.0f}% checkpoint")
        if use_league:
            print(f"  Checkpoint workers: using league pool (skill-matched sampling)")
        else:
            print(f"  Checkpoint opponent: {'loaded' if has_checkpoint_opponent else 'not available (using random)'}")
    else:
        print(f"Opponent mix: {config.get('self_play_ratio', 0.7)*100:.0f}% self-play, "
              f"{config.get('greedy_opponent_ratio', 0.2)*100:.0f}% greedy, "
              f"{config.get('random_opponent_ratio', 0.1)*100:.0f}% random")
    print("=" * 60 + "\n")

    # Create worker pool
    if num_workers > 0:
        pool = mp.Pool(processes=num_workers)
        print(f"Created worker pool with {num_workers} processes\n")

    # Training loop
    try:
        for episode in range(start_episode, total_episodes, num_workers if num_workers > 0 else 1):
            episode_start = time.time()

            # Compute epsilon using flexible schedule
            epsilon = compute_epsilon(
                episode, total_episodes,
                epsilon_start=config["epsilon_start"],
                epsilon_end=config["epsilon_end"],
                warmup_episodes=config.get("epsilon_warmup_episodes", 1000),
                schedule=config.get("epsilon_schedule", "cosine")
            )

            # Play episodes in parallel with mixed opponents
            if num_workers > 0:
                # Get CPU model state dict for workers
                cpu_model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}

                # Get opponent mix based on curriculum or fixed ratios
                if config.get("use_curriculum", False):
                    opponent_mix = get_opponent_mix(episode, total_episodes, config)
                    self_play_ratio = opponent_mix["self_play"]
                    greedy_ratio = opponent_mix["greedy"]
                    checkpoint_ratio = opponent_mix["checkpoint"]
                else:
                    self_play_ratio = config.get("self_play_ratio", 0.7)
                    greedy_ratio = config.get("greedy_opponent_ratio", 0.2)
                    checkpoint_ratio = config.get("random_opponent_ratio", 0.1)

                # Distribute workers
                n_self_play = max(1, int(num_workers * self_play_ratio))
                n_greedy = int(num_workers * greedy_ratio)
                n_checkpoint = num_workers - n_self_play - n_greedy

                # Create worker arguments for each type (include PTIE flag)
                base_args = lambda: (cpu_model_state_dict, epsilon, random.randint(0, 1000000), use_ptie)

                # Self-play workers
                self_play_args = [base_args() for _ in range(n_self_play)]

                # Greedy opponent workers
                greedy_args = [base_args() for _ in range(n_greedy)]

                # Play self-play and greedy episodes
                self_play_results = pool.map(worker_play_episode, self_play_args) if n_self_play > 0 else []
                greedy_results = pool.map(worker_play_episode_vs_greedy, greedy_args) if n_greedy > 0 else []

                # Checkpoint opponent workers (league, single checkpoint, or random fallback)
                checkpoint_results = []
                if n_checkpoint > 0:
                    if use_league and _league is not None and len(_league) > 0:
                        # League training: sample different opponents for each worker
                        checkpoint_args = []
                        for _ in range(n_checkpoint):
                            opponent = _league.sample_opponent(current_win_rate)
                            opponent_state_dict = _league.get_opponent_state_dict(opponent)
                            checkpoint_args.append(
                                (cpu_model_state_dict, opponent_state_dict, epsilon, random.randint(0, 1000000), use_ptie)
                            )
                        checkpoint_results = pool.map(worker_play_episode_vs_checkpoint, checkpoint_args)
                    elif has_checkpoint_opponent and _checkpoint_opponent_state_dict is not None:
                        # Play against single checkpoint model (include PTIE flag)
                        checkpoint_args = [
                            (cpu_model_state_dict, _checkpoint_opponent_state_dict, epsilon, random.randint(0, 1000000), use_ptie)
                            for _ in range(n_checkpoint)
                        ]
                        checkpoint_results = pool.map(worker_play_episode_vs_checkpoint, checkpoint_args)
                    else:
                        # Fallback to random opponents
                        random_args = [base_args() for _ in range(n_checkpoint)]
                        checkpoint_results = pool.map(worker_play_episode_vs_random, random_args)

                # Add self-play transitions to buffer (all 4 players)
                for trajectories, rewards in self_play_results:
                    for player in range(4):
                        episode_return = rewards[player]
                        for transition in trajectories[player]:
                            if use_ptie and len(transition) == 4:
                                state, action, move_history, perfect_state = transition
                                buffer.push(state, action, move_history, episode_return, perfect_state)
                            else:
                                state, action, move_history = transition
                                buffer.push(state, action, move_history, episode_return)

                # Add greedy opponent transitions to buffer (only player 0)
                for trajectories, rewards in greedy_results:
                    episode_return = rewards[0]  # Model is player 0
                    for transition in trajectories[0]:
                        if use_ptie and len(transition) == 4:
                            state, action, move_history, perfect_state = transition
                            buffer.push(state, action, move_history, episode_return, perfect_state)
                        else:
                            state, action, move_history = transition
                            buffer.push(state, action, move_history, episode_return)

                # Add checkpoint/random opponent transitions to buffer (only player 0)
                for trajectories, rewards in checkpoint_results:
                    episode_return = rewards[0]  # Model is player 0
                    for transition in trajectories[0]:
                        if use_ptie and len(transition) == 4:
                            state, action, move_history, perfect_state = transition
                            buffer.push(state, action, move_history, episode_return, perfect_state)
                        else:
                            state, action, move_history = transition
                            buffer.push(state, action, move_history, episode_return)
            else:
                # Single-threaded fallback (self-play only)
                trajectories, rewards = play_episode(model, epsilon, device, collect_perfect=use_ptie)
                for player in range(4):
                    episode_return = rewards[player]
                    for transition in trajectories[player]:
                        if use_ptie and len(transition) == 4:
                            state, action, move_history, perfect_state = transition
                            buffer.push(state, action, move_history, episode_return, perfect_state)
                        else:
                            state, action, move_history = transition
                            buffer.push(state, action, move_history, episode_return)

            # Training step
            loss_value = 0.0
            critic_loss_value = 0.0
            if len(buffer) >= config["batch_size"]:
                # Sample batch WITH HISTORY (and perfect states if PTIE enabled)
                if use_ptie:
                    states, actions, move_histories, returns, perfect_states = buffer.sample_arrays(
                        config["batch_size"], include_perfect=True
                    )
                else:
                    states, actions, move_histories, returns = buffer.sample_arrays(config["batch_size"])
                    perfect_states = None

                # Convert to tensors
                states_t = torch.from_numpy(states).to(device)
                actions_t = torch.from_numpy(actions).to(device)
                histories_t = torch.from_numpy(move_histories).to(device)
                returns_t = torch.from_numpy(returns).to(device)

                # PTIE: Train critic on perfect information
                if use_ptie and perfect_states is not None and critic is not None:
                    perfect_states_t = torch.from_numpy(perfect_states).to(device)

                    # Critic predicts value from perfect state
                    critic_values = critic.predict_value(perfect_states_t)

                    # Critic loss: MSE between predicted value and actual return
                    critic_loss = F.mse_loss(critic_values, returns_t)
                    critic_loss_value = critic_loss.item()

                    # Update critic
                    critic_optimizer.zero_grad()
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(critic.parameters(), config["grad_clip"])
                    critic_optimizer.step()

                # Forward pass for actor
                if isinstance(model, LSTMNetwork):
                    q_values = model.predict_q_values(histories_t, states_t, actions_t)
                else:
                    q_values = model.predict_q_values(states_t, actions_t)

                # Compute actor loss
                if use_ptie and perfect_states is not None:
                    # Use baseline-subtracted returns (advantage-weighted)
                    with torch.no_grad():
                        baseline = critic.predict_value(perfect_states_t)
                    # Actor learns to predict advantage (return - baseline)
                    actor_loss = F.mse_loss(q_values, returns_t - baseline)
                else:
                    # Standard Q-learning loss
                    actor_loss = F.mse_loss(q_values, returns_t)

                loss = actor_loss
                loss_value = loss.item()

                # Backward pass for actor
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
                optimizer.step()

            # Update target network
            if episode > 0 and episode % config["target_update_freq"] == 0:
                soft_update(target_model, model, config["tau"])

            # Evaluation
            if episode > 0 and episode % config["eval_freq"] == 0:
                eval_start = time.time()

                # Evaluate vs greedy bot opponents
                win_rate_greedy = evaluate_vs_greedy_bot(model, config["eval_games"], device)

                # Update current win rate for skill-matched league sampling
                current_win_rate = win_rate_greedy

                eval_time = time.time() - eval_start

                # Logging
                episode_time = time.time() - episode_start
                episodes_per_sec = num_workers / episode_time if num_workers > 0 else 1.0 / episode_time
                if use_ptie:
                    print(f"Episode {episode:6d} | "
                          f"ε={epsilon:.3f} | "
                          f"WinGreedy={win_rate_greedy*100:5.1f}% | "
                          f"ActorL={loss_value:.4f} | "
                          f"CriticL={critic_loss_value:.4f} | "
                          f"Buffer={len(buffer):5d} | "
                          f"Eps/s={episodes_per_sec:.1f}")
                else:
                    print(f"Episode {episode:6d} | "
                          f"ε={epsilon:.3f} | "
                          f"WinGreedy={win_rate_greedy*100:5.1f}% | "
                          f"Loss={loss_value:.4f} | "
                          f"Buffer={len(buffer):5d} | "
                          f"Eps/s={episodes_per_sec:.1f}")

                # Save best model (based on greedy bot win rate)
                # Tie-breaker: prefer newer model (higher episode) if win rates equal
                if checkpoint_path and win_rate_greedy >= best_win_rate:
                    best_win_rate = win_rate_greedy
                    best_path = checkpoint_path.replace(".pt", "_best.pt")
                    checkpoint_dict = {
                        "episode": episode,
                        "model_state_dict": model.state_dict(),
                        "target_model_state_dict": target_model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epsilon": epsilon,
                        "win_rate_greedy": win_rate_greedy,
                        "best_win_rate": best_win_rate,
                        "use_ptie": use_ptie,
                    }
                    # Include critic if PTIE is enabled
                    if use_ptie and critic is not None:
                        checkpoint_dict["critic_state_dict"] = critic.state_dict()
                        checkpoint_dict["critic_optimizer_state_dict"] = critic_optimizer.state_dict()
                    torch.save(checkpoint_dict, best_path)
                    print(f"  → Saved best model (greedy win rate: {win_rate_greedy*100:.1f}%)")

                # Top-K checkpoint management
                if checkpoint_path and win_rate_greedy > 0:
                    top_ckpt_path = checkpoint_path.replace(".pt", f"_top_{episode}.pt")

                    # Save this checkpoint
                    checkpoint_dict = {
                        "episode": episode,
                        "model_state_dict": model.state_dict(),
                        "target_model_state_dict": target_model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epsilon": epsilon,
                        "win_rate_greedy": win_rate_greedy,
                        "best_win_rate": best_win_rate,
                        "use_ptie": use_ptie,
                    }
                    if use_ptie and critic is not None:
                        checkpoint_dict["critic_state_dict"] = critic.state_dict()
                        checkpoint_dict["critic_optimizer_state_dict"] = critic_optimizer.state_dict()
                    torch.save(checkpoint_dict, top_ckpt_path)
                    top_checkpoints.append((win_rate_greedy, episode, top_ckpt_path))

                    # Sort by win rate (descending), then by episode (descending) for tie-breaking
                    # This ensures newer models are preferred when win rates are equal
                    top_checkpoints.sort(key=lambda x: (x[0], x[1]), reverse=True)

                    # Remove excess checkpoints (keep only top K)
                    while len(top_checkpoints) > top_k:
                        _, _, old_path = top_checkpoints.pop()
                        if os.path.exists(old_path):
                            os.remove(old_path)

            # Save checkpoint
            if checkpoint_path and episode > 0 and episode % config["save_freq"] == 0:
                checkpoint_dict = {
                    "episode": episode,
                    "model_state_dict": model.state_dict(),
                    "target_model_state_dict": target_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epsilon": epsilon,
                    "best_win_rate": best_win_rate,
                    "use_ptie": use_ptie,
                }
                # Include critic if PTIE is enabled
                if use_ptie and critic is not None:
                    checkpoint_dict["critic_state_dict"] = critic.state_dict()
                    checkpoint_dict["critic_optimizer_state_dict"] = critic_optimizer.state_dict()
                torch.save(checkpoint_dict, checkpoint_path)

            # League snapshot
            if use_league and _league is not None:
                snapshot_freq = config.get("league_snapshot_freq", 2000)
                if episode > 0 and episode % snapshot_freq == 0:
                    cpu_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
                    snapshot_path = _league.add_snapshot(cpu_state_dict, episode, current_win_rate)
                    print(f"  → League snapshot saved (gen {episode}, win_rate={current_win_rate*100:.1f}%, pool size={len(_league)})")
    finally:
        # Clean up worker pool
        if num_workers > 0:
            pool.close()
            pool.join()
            print("\nWorker pool closed")

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best win rate: {best_win_rate*100:.1f}%")
    print(f"Final epsilon: {epsilon:.3f}")
    print(f"Buffer size: {len(buffer)}")
    print("=" * 60 + "\n")

    return model, best_win_rate


if __name__ == "__main__":
    # Quick test
    print("Testing training components (Stage 2)...")

    # Test with network based on config
    if NETWORK_CONFIG.get("use_lstm", False):
        model = LSTMNetwork(**NETWORK_CONFIG)
        print("Testing with LSTM Network")
    else:
        model = SimpleNetwork(**NETWORK_CONFIG)
        print("Testing with Simple Network")

    device = "cpu"

    # Test action selection
    game = Big2Game(seed=42)
    current_player = game.current_player
    action_idx, legal_moves = select_action(game, current_player, model, 0.5, device)
    print(f"Selected action {action_idx} from {len(legal_moves)} legal moves for player {current_player}")

    # Test episode
    trajectories, rewards = play_episode(model, 0.5, device)
    print(f"Episode: {len(trajectories[0])} transitions for player 0")
    print(f"First transition has state, action, and move_history")
    print(f"Rewards: {rewards}")

    print("\nTest complete!")
