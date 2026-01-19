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

from ..env import Big2Game, encode_state, encode_action, get_legal_moves
from ..models import SimpleNetwork
from ..config import TRAINING_CONFIG, NETWORK_CONFIG
from .buffer import ReplayBuffer
from ..agents import select_action_greedy_bot


def select_action(
    game: Big2Game,
    player: int,
    model: SimpleNetwork,
    epsilon: float,
    device: str
) -> Tuple[int, List]:
    """
    Select an action using epsilon-greedy policy.

    Args:
        game: Current game state
        player: Player index
        model: Policy network
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

    # Evaluate all legal actions
    state = encode_state(game, player)
    state_tensor = torch.from_numpy(state).unsqueeze(0).to(device)

    q_values = []
    with torch.no_grad():
        for move in legal_moves:
            action = encode_action(move)
            action_tensor = torch.from_numpy(action).unsqueeze(0).to(device)
            x = torch.cat([state_tensor, action_tensor], dim=1)
            q = model(x)
            q_values.append(q.item())

    # Select action with highest Q-value
    best_action = int(np.argmax(q_values))
    return best_action, legal_moves


def play_episode(
    model: SimpleNetwork,
    epsilon: float,
    device: str
) -> Tuple[List[List[Tuple]], List[float]]:
    """
    Play one episode of self-play.

    Args:
        model: Policy network
        epsilon: Exploration rate
        device: Device to run model on

    Returns:
        (trajectories, rewards) where:
        - trajectories: List of 4 player trajectories, each containing (state, action) tuples
        - rewards: List of 4 final rewards
    """
    game = Big2Game()
    trajectories = [[], [], [], []]

    while not game.done:
        player = game.current_player

        # Get state
        state = encode_state(game, player)

        # Select action
        action_idx, legal_moves = select_action(game, player, model, epsilon, device)
        move = legal_moves[action_idx]

        # Encode action
        action_enc = encode_action(move)

        # Store in trajectory
        trajectories[player].append((state.copy(), action_enc.copy()))

        # Step game
        _, _, done, info = game.step(move)

    # Get final rewards
    rewards = info["all_rewards"]

    return trajectories, rewards


def soft_update(target_net: SimpleNetwork, source_net: SimpleNetwork, tau: float):
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
    model: SimpleNetwork,
    num_games: int,
    device: str
) -> float:
    """
    Evaluate model against random opponents.

    Args:
        model: Policy network
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
    model: SimpleNetwork,
    num_games: int,
    device: str
) -> float:
    """
    Evaluate model against greedy bot opponents.

    Args:
        model: Policy network
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
    Worker function to play one episode.

    This function is designed to be called by multiprocessing workers.

    Args:
        args: Tuple of (model_state_dict, epsilon, seed)

    Returns:
        (trajectories, rewards) where:
        - trajectories: List of 4 player trajectories
        - rewards: List of 4 final rewards
    """
    model_state_dict, epsilon, seed = args

    # Set random seed for this worker
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    # Create model on CPU (workers use CPU only)
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

            # Get state
            state = encode_state(game, player)

            # Select action
            action_idx, legal_moves = select_action(game, player, model, epsilon, device)
            move = legal_moves[action_idx]

            # Encode action
            action_enc = encode_action(move)

            # Store in trajectory
            trajectories[player].append((state.copy(), action_enc.copy()))

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

    # Create model
    model = SimpleNetwork(**NETWORK_CONFIG).to(device)
    target_model = copy.deepcopy(model)
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
    print("=" * 60 + "\n")

    # Create worker pool
    if num_workers > 0:
        pool = mp.Pool(processes=num_workers)
        print(f"Created worker pool with {num_workers} processes\n")

    # Training loop
    try:
        for episode in range(start_episode, config["num_episodes"], num_workers if num_workers > 0 else 1):
            episode_start = time.time()

            # Play episodes in parallel
            if num_workers > 0:
                # Get CPU model state dict for workers
                cpu_model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}

                # Create worker arguments
                worker_args = [
                    (cpu_model_state_dict, epsilon, random.randint(0, 1000000))
                    for _ in range(num_workers)
                ]

                # Play episodes in parallel
                results = pool.map(worker_play_episode, worker_args)

                # Add all transitions to buffer
                for trajectories, rewards in results:
                    for player in range(4):
                        episode_return = rewards[player]
                        for state, action in trajectories[player]:
                            buffer.push(state, action, episode_return)
            else:
                # Single-threaded fallback
                trajectories, rewards = play_episode(model, epsilon, device)
                for player in range(4):
                    episode_return = rewards[player]
                    for state, action in trajectories[player]:
                        buffer.push(state, action, episode_return)

            # Decay epsilon (once per batch of episodes)
            epsilon = max(config["epsilon_end"], epsilon * config["epsilon_decay"])

            # Training step
            loss_value = 0.0
            if len(buffer) >= config["batch_size"]:
                # Sample batch
                states, actions, returns = buffer.sample_arrays(config["batch_size"])

                # Convert to tensors
                states_t = torch.from_numpy(states).to(device)
                actions_t = torch.from_numpy(actions).to(device)
                returns_t = torch.from_numpy(returns).to(device)

                # Forward pass
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
    print("Testing training components...")
    model = SimpleNetwork()
    device = "cpu"

    # Test action selection
    game = Big2Game(seed=42)
    current_player = game.current_player
    action_idx, legal_moves = select_action(game, current_player, model, 0.5, device)
    print(f"Selected action {action_idx} from {len(legal_moves)} legal moves for player {current_player}")

    # Test episode
    trajectories, rewards = play_episode(model, 0.5, device)
    print(f"Episode: {len(trajectories[0])} transitions for player 0, rewards: {rewards}")

    print("\nTest complete!")
