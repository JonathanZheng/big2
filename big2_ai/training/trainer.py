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

from ..env import Big2Game, encode_state, encode_action, get_legal_moves, encode_move_history
from ..models import SimpleNetwork, LSTMNetwork
from ..config import TRAINING_CONFIG, NETWORK_CONFIG
from .buffer import ReplayBuffer
from ..agents import select_action_greedy_bot


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
    device: str
) -> Tuple[List[List[Tuple]], List[float]]:
    """
    Play one episode of self-play (Stage 2).

    Args:
        model: Policy network (SimpleNetwork or LSTMNetwork)
        epsilon: Exploration rate
        device: Device to run model on

    Returns:
        (trajectories, rewards) where:
        - trajectories: List of 4 player trajectories, each containing (state, action, move_history) tuples
        - rewards: List of 4 final rewards
    """
    game = Big2Game()
    trajectories = [[], [], [], []]

    while not game.done:
        player = game.current_player

        # Encode state
        state = encode_state(game, player)

        # Encode move history
        move_history = encode_move_history(game, max_moves=16)

        # Select action
        action_idx, legal_moves = select_action(game, player, model, epsilon, device)
        move = legal_moves[action_idx]

        # Encode action
        action_enc = encode_action(move)

        # Store in trajectory WITH move history
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
    Worker function to play one episode (Stage 2).

    This function is designed to be called by multiprocessing workers.

    Args:
        args: Tuple of (model_state_dict, epsilon, seed)

    Returns:
        (trajectories, rewards) where:
        - trajectories: List of 4 player trajectories, each containing (state, action, move_history) tuples
        - rewards: List of 4 final rewards
    """
    model_state_dict, epsilon, seed = args

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

    # Play episode
    game = Big2Game()
    trajectories = [[], [], [], []]

    with torch.no_grad():
        while not game.done:
            player = game.current_player

            # Encode state
            state = encode_state(game, player)

            # Encode move history
            move_history = encode_move_history(game, max_moves=16)

            # Select action
            action_idx, legal_moves = select_action(game, player, model, epsilon, device)
            move = legal_moves[action_idx]

            # Encode action
            action_enc = encode_action(move)

            # Store in trajectory WITH move history
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
        args: Tuple of (model_state_dict, epsilon, seed)

    Returns:
        (trajectories, rewards) where:
        - trajectories: List of 4 player trajectories (only player 0 has data)
        - rewards: List of 4 final rewards
    """
    model_state_dict, epsilon, seed = args

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

    # Play episode
    game = Big2Game()
    trajectories = [[], [], [], []]  # Only player 0 will be populated

    with torch.no_grad():
        while not game.done:
            player = game.current_player

            if player == 0:
                # Model plays as player 0
                state = encode_state(game, player)
                move_history = encode_move_history(game, max_moves=16)
                action_idx, legal_moves = select_action(game, player, model, epsilon, device)
                move = legal_moves[action_idx]
                action_enc = encode_action(move)

                # Store in trajectory
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
        args: Tuple of (model_state_dict, epsilon, seed)

    Returns:
        (trajectories, rewards) where:
        - trajectories: List of 4 player trajectories (only player 0 has data)
        - rewards: List of 4 final rewards
    """
    model_state_dict, epsilon, seed = args

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

    # Play episode
    game = Big2Game()
    trajectories = [[], [], [], []]  # Only player 0 will be populated

    with torch.no_grad():
        while not game.done:
            player = game.current_player

            if player == 0:
                # Model plays as player 0
                state = encode_state(game, player)
                move_history = encode_move_history(game, max_moves=16)
                action_idx, legal_moves = select_action(game, player, model, epsilon, device)
                move = legal_moves[action_idx]
                action_enc = encode_action(move)

                # Store in trajectory
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


def train(
    num_episodes: Optional[int] = None,
    checkpoint_path: Optional[str] = None,
    resume: bool = False,
    num_workers: int = 6
):
    """
    Main training loop with parallel workers.

    Args:
        num_episodes: Number of episodes to train (overrides config)
        checkpoint_path: Path to save checkpoints
        resume: Whether to resume from checkpoint
        num_workers: Number of parallel workers (default: 6)
    """
    # Get config
    config = TRAINING_CONFIG
    if num_episodes is not None:
        config["num_episodes"] = num_episodes

    # Setup device
    if torch.backends.mps.is_available():
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

    # Create replay buffer
    buffer = ReplayBuffer(capacity=config["buffer_size"])

    # Initialize training state
    start_episode = 0
    epsilon = config["epsilon_start"]
    best_win_rate = 0.0

    # Resume from checkpoint if requested
    if resume and checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        target_model.load_state_dict(checkpoint["target_model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_episode = checkpoint["episode"]
        epsilon = checkpoint["epsilon"]
        best_win_rate = checkpoint.get("best_win_rate", 0.0)

    # Create checkpoint directory
    if checkpoint_path:
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)
    print(f"Episodes: {config['num_episodes']}")
    print(f"Workers: {num_workers}")
    print(f"Buffer size: {config['buffer_size']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Learning rate: {config['learning_rate']}")
    print(f"Epsilon: {epsilon:.3f} → {config['epsilon_end']:.3f}")
    print(f"Opponent mix: {config.get('self_play_ratio', 0.7)*100:.0f}% self-play, "
          f"{config.get('greedy_opponent_ratio', 0.2)*100:.0f}% greedy, "
          f"{config.get('random_opponent_ratio', 0.1)*100:.0f}% random")
    print("=" * 60 + "\n")

    # Create worker pool
    if num_workers > 0:
        pool = mp.Pool(processes=num_workers)
        print(f"Created worker pool with {num_workers} processes\n")

    # Track win rates for adaptive epsilon
    recent_win_rates = []

    # Training loop
    try:
        for episode in range(start_episode, config["num_episodes"], num_workers if num_workers > 0 else 1):
            episode_start = time.time()

            # Play episodes in parallel with mixed opponents
            if num_workers > 0:
                # Get CPU model state dict for workers
                cpu_model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}

                # Calculate worker distribution based on opponent diversity ratios
                self_play_ratio = config.get("self_play_ratio", 0.7)
                greedy_ratio = config.get("greedy_opponent_ratio", 0.2)
                random_ratio = config.get("random_opponent_ratio", 0.1)

                # Distribute workers (use floor for each type, remainder goes to self-play)
                n_self_play = max(1, int(num_workers * self_play_ratio))
                n_greedy = int(num_workers * greedy_ratio)
                n_random = num_workers - n_self_play - n_greedy

                # Create worker arguments for each type
                base_args = lambda: (cpu_model_state_dict, epsilon, random.randint(0, 1000000))

                # Self-play workers
                self_play_args = [base_args() for _ in range(n_self_play)]

                # Greedy opponent workers
                greedy_args = [base_args() for _ in range(n_greedy)]

                # Random opponent workers
                random_args = [base_args() for _ in range(n_random)]

                # Play episodes in parallel using starmap_async to run different worker functions
                self_play_results = pool.map(worker_play_episode, self_play_args) if n_self_play > 0 else []
                greedy_results = pool.map(worker_play_episode_vs_greedy, greedy_args) if n_greedy > 0 else []
                random_results = pool.map(worker_play_episode_vs_random, random_args) if n_random > 0 else []

                # Add self-play transitions to buffer (all 4 players)
                for trajectories, rewards in self_play_results:
                    for player in range(4):
                        episode_return = rewards[player]
                        for state, action, move_history in trajectories[player]:
                            buffer.push(state, action, move_history, episode_return)

                # Add greedy opponent transitions to buffer (only player 0)
                for trajectories, rewards in greedy_results:
                    episode_return = rewards[0]  # Model is player 0
                    for state, action, move_history in trajectories[0]:
                        buffer.push(state, action, move_history, episode_return)

                # Add random opponent transitions to buffer (only player 0)
                for trajectories, rewards in random_results:
                    episode_return = rewards[0]  # Model is player 0
                    for state, action, move_history in trajectories[0]:
                        buffer.push(state, action, move_history, episode_return)
            else:
                # Single-threaded fallback (self-play only)
                trajectories, rewards = play_episode(model, epsilon, device)
                for player in range(4):
                    episode_return = rewards[player]
                    for state, action, move_history in trajectories[player]:
                        buffer.push(state, action, move_history, episode_return)

            # Decay epsilon (once per batch of episodes)
            epsilon = max(config["epsilon_end"], epsilon * config["epsilon_decay"])

            # Training step
            loss_value = 0.0
            if len(buffer) >= config["batch_size"]:
                # Sample batch WITH HISTORY
                states, actions, move_histories, returns = buffer.sample_arrays(config["batch_size"])

                # Convert to tensors
                states_t = torch.from_numpy(states).to(device)
                actions_t = torch.from_numpy(actions).to(device)
                histories_t = torch.from_numpy(move_histories).to(device)
                returns_t = torch.from_numpy(returns).to(device)

                # Forward pass
                if isinstance(model, LSTMNetwork):
                    q_values = model.predict_q_values(histories_t, states_t, actions_t)
                else:
                    q_values = model.predict_q_values(states_t, actions_t)

                # Compute loss
                loss = F.mse_loss(q_values, returns_t)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
                optimizer.step()

                loss_value = loss.item()

            # Update target network
            if episode > 0 and episode % config["target_update_freq"] == 0:
                soft_update(target_model, model, config["tau"])

            # Evaluation
            if episode > 0 and episode % config["eval_freq"] == 0:
                eval_start = time.time()

                # Evaluate vs random opponents
                win_rate_random = evaluate_vs_random(model, config["eval_games"], device)

                # Evaluate vs greedy bot opponents
                win_rate_greedy = evaluate_vs_greedy_bot(model, config["eval_games"], device)

                eval_time = time.time() - eval_start

                # Adaptive epsilon scheduling
                if config.get("adaptive_epsilon", False):
                    recent_win_rates.append(win_rate_random)
                    if len(recent_win_rates) > 5:
                        recent_win_rates.pop(0)

                    # If win rate drops, increase epsilon to explore more
                    if len(recent_win_rates) >= 2:
                        win_rate_change = recent_win_rates[-1] - recent_win_rates[-2]
                        if win_rate_change < -config["epsilon_adapt_threshold"]:
                            epsilon = min(config["epsilon_start"], epsilon * 1.1)
                            print(f"  → Increased epsilon to {epsilon:.3f} (win rate dropped by {-win_rate_change*100:.1f}%)")

                # Logging
                episode_time = time.time() - episode_start
                episodes_per_sec = num_workers / episode_time if num_workers > 0 else 1.0 / episode_time
                print(f"Episode {episode:6d} | "
                      f"ε={epsilon:.3f} | "
                      f"WinRand={win_rate_random*100:5.1f}% | "
                      f"WinGreedy={win_rate_greedy*100:5.1f}% | "
                      f"Loss={loss_value:.4f} | "
                      f"Buffer={len(buffer):5d} | "
                      f"Eps/s={episodes_per_sec:.1f}")

                # Save best model (based on greedy bot win rate)
                if checkpoint_path and win_rate_greedy > best_win_rate:
                    best_win_rate = win_rate_greedy
                    best_path = checkpoint_path.replace(".pt", "_best.pt")
                    torch.save({
                        "episode": episode,
                        "model_state_dict": model.state_dict(),
                        "target_model_state_dict": target_model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epsilon": epsilon,
                        "win_rate_random": win_rate_random,
                        "win_rate_greedy": win_rate_greedy,
                        "best_win_rate": best_win_rate,
                    }, best_path)
                    print(f"  → Saved best model (greedy win rate: {win_rate_greedy*100:.1f}%, random win rate: {win_rate_random*100:.1f}%)")

            # Save checkpoint
            if checkpoint_path and episode > 0 and episode % config["save_freq"] == 0:
                torch.save({
                    "episode": episode,
                    "model_state_dict": model.state_dict(),
                    "target_model_state_dict": target_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epsilon": epsilon,
                    "best_win_rate": best_win_rate,
                }, checkpoint_path)
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
