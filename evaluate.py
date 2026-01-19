"""Evaluation script for Big 2 AI."""

import argparse
import torch
import random
import numpy as np
from collections import defaultdict
from typing import List

from big2_ai.env import Big2Game, encode_state, encode_action, get_legal_moves
from big2_ai.models import SimpleNetwork
from big2_ai.config import NETWORK_CONFIG
from big2_ai.agents import select_action_greedy_bot


def select_action_greedy(game: Big2Game, player: int, model: SimpleNetwork, device: str):
    """Select action greedily (no exploration)."""
    legal_moves = get_legal_moves(game, player)

    if len(legal_moves) == 0:
        raise ValueError(f"No legal moves for player {player}")

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

    best_action = int(np.argmax(q_values))
    return legal_moves[best_action]


# =============================================================================
# Game Play Functions
# =============================================================================

def play_game(agents, device: str, verbose: bool = False):
    """
    Play a game with given agents.

    Args:
        agents: List of 4 agents (model, 'random', or 'greedy_bot')
        device: Device to run model on
        verbose: Whether to print game progress

    Returns:
        Winner index (0-3)
    """
    game = Big2Game()

    if verbose:
        print("\n" + "=" * 60)
        print("Starting Game")
        print("=" * 60)
        print(game)
        print()

    move_count = 0

    while not game.done:
        player = game.current_player
        agent = agents[player]

        if agent == "random":
            legal_moves = get_legal_moves(game, player)
            move = random.choice(legal_moves)
        elif agent == "greedy_bot":
            move = select_action_greedy_bot(game, player)
        else:
            move = select_action_greedy(game, player, agent, device)

        if verbose:
            if agent == "random":
                agent_name = "random"
            elif agent == "greedy_bot":
                agent_name = "greedy_bot"
            else:
                agent_name = type(agent).__name__
            print(f"Player {player} ({agent_name}): {move}")

        _, _, done, info = game.step(move)
        move_count += 1

        if verbose and done:
            print(f"\nGame over! Winner: Player {info['winner']}")
            print(f"Total moves: {move_count}")
            print(f"Rewards: {info['all_rewards']}")

    return info["winner"]


def evaluate(model_path: str, num_games: int = 100, verbose: bool = False):
    """
    Evaluate trained model.

    Args:
        model_path: Path to model checkpoint
        num_games: Number of games to evaluate
        verbose: Whether to print detailed results
    """
    # Setup device
    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # Load model
    print(f"Loading model from: {model_path}")
    model = SimpleNetwork(**NETWORK_CONFIG).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Episode: {checkpoint.get('episode', 'unknown')}")
        print(f"Best win rate: {checkpoint.get('best_win_rate', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    print(f"\nEvaluating over {num_games} games...")
    print("=" * 60)

    # Evaluate against random opponents
    print("\n1. Model vs 3 Random Opponents")
    print("-" * 60)

    wins = defaultdict(int)
    for i in range(num_games):
        agents = [model, "random", "random", "random"]
        winner = play_game(agents, device, verbose=verbose and i == 0)
        wins[winner] += 1

        if (i + 1) % 20 == 0:
            print(f"  Progress: {i+1}/{num_games} games")

    print(f"\nResults:")
    print(f"  Model wins:     {wins[0]:3d} ({wins[0]/num_games*100:5.1f}%)")
    print(f"  Opponent wins:  {num_games - wins[0]:3d} ({(num_games-wins[0])/num_games*100:5.1f}%)")
    print(f"    Player 1:     {wins[1]:3d} ({wins[1]/num_games*100:5.1f}%)")
    print(f"    Player 2:     {wins[2]:3d} ({wins[2]/num_games*100:5.1f}%)")
    print(f"    Player 3:     {wins[3]:3d} ({wins[3]/num_games*100:5.1f}%)")

    # Evaluate against greedy bot opponents
    print("\n2. Model vs 3 Greedy Bot Opponents")
    print("-" * 60)

    wins_greedy = defaultdict(int)
    for i in range(num_games):
        agents = [model, "greedy_bot", "greedy_bot", "greedy_bot"]
        winner = play_game(agents, device, verbose=False)
        wins_greedy[winner] += 1

        if (i + 1) % 20 == 0:
            print(f"  Progress: {i+1}/{num_games} games")

    print(f"\nResults:")
    print(f"  Model wins:     {wins_greedy[0]:3d} ({wins_greedy[0]/num_games*100:5.1f}%)")
    print(f"  Opponent wins:  {num_games - wins_greedy[0]:3d} ({(num_games-wins_greedy[0])/num_games*100:5.1f}%)")
    print(f"    Player 1:     {wins_greedy[1]:3d} ({wins_greedy[1]/num_games*100:5.1f}%)")
    print(f"    Player 2:     {wins_greedy[2]:3d} ({wins_greedy[2]/num_games*100:5.1f}%)")
    print(f"    Player 3:     {wins_greedy[3]:3d} ({wins_greedy[3]/num_games*100:5.1f}%)")

    # Model in different positions
    print("\n3. Model Performance by Position")
    print("-" * 60)

    for pos in range(4):
        agents = ["random", "random", "random", "random"]
        agents[pos] = model

        wins = 0
        for _ in range(num_games):
            winner = play_game(agents, device)
            if winner == pos:
                wins += 1

        print(f"  Position {pos}: {wins:3d}/{num_games} ({wins/num_games*100:5.1f}%)")

    print("\n" + "=" * 60)


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate Big 2 AI")

    parser.add_argument(
        "checkpoint",
        type=str,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--games",
        type=int,
        default=100,
        help="Number of games to evaluate (default: 100)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed game information"
    )

    args = parser.parse_args()

    evaluate(args.checkpoint, args.games, args.verbose)


if __name__ == "__main__":
    main()
