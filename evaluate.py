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
from big2_ai.agents import select_action_greedy_bot, select_action_rule_based_bot


def load_model(model_path: str, device: str):
    """Load a model from checkpoint."""
    model = SimpleNetwork(**NETWORK_CONFIG).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        episode = checkpoint.get('episode', 'unknown')
    else:
        model.load_state_dict(checkpoint)
        episode = 'unknown'
    model.eval()
    return model, episode


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

def play_game(agents, device: str, verbose: bool = False, track_passes: bool = False):
    """
    Play a game with given agents.

    Args:
        agents: List of 4 agents (model, 'random', 'greedy_bot', or 'rule_based_bot')
        device: Device to run model on
        verbose: Whether to print game progress
        track_passes: Whether to track pass statistics

    Returns:
        Winner index (0-3) if not tracking passes
        (winner, pass_stats) tuple if tracking passes
    """
    game = Big2Game()

    if verbose:
        print("\n" + "=" * 60)
        print("Starting Game")
        print("=" * 60)
        print(game)
        print()

    move_count = 0

    # Pass tracking stats per player
    pass_stats = {i: {"total_moves": 0, "total_passes": 0, "passes_with_alternatives": 0} for i in range(4)}

    while not game.done:
        player = game.current_player
        agent = agents[player]

        legal_moves = get_legal_moves(game, player)

        if agent == "random":
            move = random.choice(legal_moves)
        elif agent == "greedy_bot":
            move = select_action_greedy_bot(game, player)
        elif agent == "rule_based_bot":
            move = select_action_rule_based_bot(game, player)
        else:
            move = select_action_greedy(game, player, agent, device)

        # Track pass statistics
        if track_passes:
            pass_stats[player]["total_moves"] += 1
            if move.is_pass():
                pass_stats[player]["total_passes"] += 1
                # Count non-pass alternatives
                non_pass_alternatives = sum(1 for m in legal_moves if not m.is_pass())
                if non_pass_alternatives > 0:
                    pass_stats[player]["passes_with_alternatives"] += 1

        if verbose:
            if agent == "random":
                agent_name = "random"
            elif agent == "greedy_bot":
                agent_name = "greedy_bot"
            elif agent == "rule_based_bot":
                agent_name = "rule_based_bot"
            else:
                agent_name = type(agent).__name__
            print(f"Player {player} ({agent_name}): {move}")

        _, _, done, info = game.step(move)
        move_count += 1

        if verbose and done:
            print(f"\nGame over! Winner: Player {info['winner']}")
            print(f"Total moves: {move_count}")
            print(f"Rewards: {info['all_rewards']}")

    if track_passes:
        return info["winner"], pass_stats
    return info["winner"]


def run_pass_diagnosis(num_games: int = 100, model_path: str = None):
    """
    Run pass diagnosis on agents.

    Analyzes how often agents pass when they have other legal moves available.

    Args:
        num_games: Number of games to analyze
        model_path: Optional path to model checkpoint. If provided, model plays as player 0.
    """
    # Setup device
    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    model = None
    if model_path:
        print(f"Loading model from: {model_path}")
        model = SimpleNetwork(**NETWORK_CONFIG).to(device)
        checkpoint = torch.load(model_path, map_location=device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Episode: {checkpoint.get('episode', 'unknown')}")
        else:
            model.load_state_dict(checkpoint)
        model.eval()

    agent_desc = "model vs greedy_bot" if model else "greedy_bot agents"
    print(f"\nPass Diagnosis ({num_games} games, {agent_desc})")
    print("=" * 60)

    # Aggregate stats across all games
    aggregate_stats = {i: {"total_moves": 0, "total_passes": 0, "passes_with_alternatives": 0} for i in range(4)}

    for game_num in range(num_games):
        if model:
            agents = [model, "greedy_bot", "greedy_bot", "greedy_bot"]
        else:
            agents = ["greedy_bot", "greedy_bot", "greedy_bot", "greedy_bot"]
        _, pass_stats = play_game(agents, device=device, track_passes=True)

        # Aggregate
        for player in range(4):
            for key in aggregate_stats[player]:
                aggregate_stats[player][key] += pass_stats[player][key]

        if (game_num + 1) % 20 == 0:
            print(f"  Progress: {game_num + 1}/{num_games} games")

    print()

    # Print per-player stats
    total_passes_with_alt = 0
    total_passes = 0

    for player in range(4):
        stats = aggregate_stats[player]
        total_moves = stats["total_moves"]
        total_p = stats["total_passes"]
        passes_alt = stats["passes_with_alternatives"]

        total_passes += total_p
        total_passes_with_alt += passes_alt

        pass_pct = (total_p / total_moves * 100) if total_moves > 0 else 0
        alt_pct = (passes_alt / total_p * 100) if total_p > 0 else 0

        agent_label = "Model" if (model and player == 0) else "Greedy Bot"
        print(f"Player {player} ({agent_label}):")
        print(f"  Total moves: {total_moves}")
        print(f"  Total passes: {total_p} ({pass_pct:.1f}%)")
        print(f"  Passes with alternatives: {passes_alt} ({alt_pct:.1f}% of passes)")
        print()

    # Overall stats
    overall_alt_pct = (total_passes_with_alt / total_passes * 100) if total_passes > 0 else 0
    print("Overall:")
    print(f"  Total passes with alternatives: {total_passes_with_alt} / {total_passes} ({overall_alt_pct:.1f}%)")
    print("=" * 60)


def run_bot_vs_bot_evaluation(bot1_type: str, bot2_type: str, num_games: int = 100, verbose: bool = False):
    """
    Evaluate two bot types against each other.

    Args:
        bot1_type: First bot type ('greedy_bot' or 'rule_based_bot')
        bot2_type: Second bot type ('greedy_bot' or 'rule_based_bot')
        num_games: Number of games to evaluate
        verbose: Whether to print detailed results
    """
    # Setup device (not used for bots, but needed for play_game)
    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"\nBot vs Bot Evaluation ({num_games} games)")
    print("=" * 60)
    print(f"Configuration: 1x {bot1_type} vs 3x {bot2_type}")
    print()

    # Build agents list: bot1 in position 0, bot2 in positions 1-3
    agents = [bot1_type, bot2_type, bot2_type, bot2_type]

    # Track wins
    wins = defaultdict(int)

    for i in range(num_games):
        winner = play_game(agents, device, verbose=verbose and i == 0)
        wins[winner] += 1

        if (i + 1) % 20 == 0:
            print(f"  Progress: {i+1}/{num_games} games")

    print()
    print("Results:")
    print("-" * 60)

    # Sort by win rate
    results = [(player, wins[player]) for player in range(4)]
    results.sort(key=lambda x: x[1], reverse=True)

    bot1_name = bot1_type.replace("_", " ").title()
    bot2_name = bot2_type.replace("_", " ").title()

    for player, win_count in results:
        win_pct = win_count / num_games * 100
        if player == 0:
            agent_name = f"Player {player}: {bot1_name}"
        else:
            agent_name = f"Player {player}: {bot2_name}"

        print(f"  {agent_name}")
        print(f"    Wins: {win_count:3d} ({win_pct:5.1f}%)")
        print()

    # Summary
    bot1_wins = wins[0]
    bot2_wins = sum(wins[i] for i in range(1, 4))

    print("Summary:")
    print("-" * 60)
    print(f"  {bot1_name}: {bot1_wins:3d} wins ({bot1_wins/num_games*100:5.1f}%)")
    print(f"  {bot2_name}: {bot2_wins:3d} wins ({bot2_wins/num_games*100:5.1f}%)")
    print()
    print("=" * 60)


def run_multi_model_evaluation(model_paths: List[str], num_games: int = 100, verbose: bool = False, opponent_type: str = "greedy_bot"):
    """
    Evaluate multiple models against each other.

    Args:
        model_paths: List of paths to model checkpoints (1-4 models)
        num_games: Number of games to evaluate
        verbose: Whether to print detailed results
        opponent_type: Type of bot to fill remaining slots ('greedy_bot' or 'rule_based_bot')
    """
    if len(model_paths) > 4:
        raise ValueError("Maximum 4 models supported")

    # Setup device
    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # Load models
    models = []
    model_names = []
    print("Loading models...")
    print("-" * 60)
    for i, path in enumerate(model_paths):
        model, episode = load_model(path, device)
        models.append(model)
        # Extract filename for display
        name = path.split("/")[-1].replace(".pt", "")
        model_names.append(f"Model {i}: {name} (ep {episode})")
        print(f"  {model_names[-1]}")

    # Fill remaining slots with the selected bot type
    num_models = len(models)
    num_bots = 4 - num_models
    for i in range(num_bots):
        model_names.append(f"Player {num_models + i}: {opponent_type}")
        print(f"  {model_names[-1]}")

    print()
    print(f"Multi-Model Evaluation ({num_games} games)")
    print("=" * 60)

    # Build agents list using the requested opponent type
    agents = models + [opponent_type] * num_bots

    # Track wins
    wins = defaultdict(int)

    for i in range(num_games):
        winner = play_game(agents, device, verbose=verbose and i == 0)
        wins[winner] += 1

        if (i + 1) % 20 == 0:
            print(f"  Progress: {i+1}/{num_games} games")

    print()
    print("Results:")
    print("-" * 60)

    # Sort by win rate
    results = [(player, wins[player]) for player in range(4)]
    results.sort(key=lambda x: x[1], reverse=True)

    for player, win_count in results:
        win_pct = win_count / num_games * 100
        # Determine label based on player index
        if player < num_models:
            agent_type = "Model"
        else:
            # Format "rule_based_bot" -> "Rule Based Bot"
            agent_type = opponent_type.replace("_", " ").title()
            
        name = model_names[player]
        print(f"  {name}")
        print(f"    Wins: {win_count:3d} ({win_pct:5.1f}%)")
        print()

    # Head-to-head summary for models only
    if num_models > 1:
        print("Model Rankings:")
        print("-" * 60)
        model_results = [(p, wins[p]) for p in range(num_models)]
        model_results.sort(key=lambda x: x[1], reverse=True)
        for rank, (player, win_count) in enumerate(model_results, 1):
            name = model_paths[player].split("/")[-1].replace(".pt", "")
            win_pct = win_count / num_games * 100
            print(f"  {rank}. {name}: {win_count} wins ({win_pct:.1f}%)")

    print()
    print("=" * 60)


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
    print("\n2. Model vs Bot Opponents")
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
        nargs="?",
        default=None,
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
    parser.add_argument(
        "--pass_diagnosis",
        action="store_true",
        help="Analyze how often bots pass when they have other legal moves"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        help="Multiple model checkpoints to evaluate against each other (1-4 models, rest are greedy bots)"
    )
    parser.add_argument(
        "--rule-based",
        action="store_true",
        help="Use rule_based_bot instead of greedy_bot for filler opponents in multi-model evaluation"
    )
    parser.add_argument(
        "--bot-vs-bot",
        type=str,
        nargs=2,
        metavar=("BOT1", "BOT2"),
        help="Evaluate BOT1 (player 0) vs 3x BOT2 (players 1-3). Options: greedy_bot, rule_based_bot"
    )

    args = parser.parse_args()

    if args.bot_vs_bot:
        bot1, bot2 = args.bot_vs_bot
        valid_bots = ["greedy_bot", "rule_based_bot"]
        if bot1 not in valid_bots or bot2 not in valid_bots:
            parser.error(f"Invalid bot type. Must be one of: {', '.join(valid_bots)}")
        run_bot_vs_bot_evaluation(bot1, bot2, args.games, args.verbose)
    elif args.models:
        opponent = "rule_based_bot" if args.rule_based else "greedy_bot"
        run_multi_model_evaluation(args.models, args.games, args.verbose, opponent_type=opponent)
    elif args.pass_diagnosis:
        run_pass_diagnosis(args.games, args.checkpoint)
    else:
        if args.checkpoint is None:
            parser.error("checkpoint is required unless --pass_diagnosis, --models, or --bot-vs-bot is specified")
        evaluate(args.checkpoint, args.games, args.verbose)


if __name__ == "__main__":
    main()
