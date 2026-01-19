"""Interactive play script for Big 2 AI."""

import argparse
import random
import numpy as np
import torch
from typing import List, Optional

from big2_ai.env.game import Big2Game, Move, card_to_str, cards_to_str, card_rank, RANK_NAMES
from big2_ai.env.move_generator import get_legal_moves
from big2_ai.env.move_detector import detect_move_type, MoveType
from big2_ai.models import SimpleNetwork
from big2_ai.config import NETWORK_CONFIG
from evaluate import select_action_greedy


def parse_card_input(input_str: str) -> Optional[int]:
    """Parse card notation (e.g., '3d', 'Ah') to card index (0-51)."""
    # Remove whitespace
    input_str = input_str.strip().lower()

    # Parse rank (first part)
    if input_str.startswith('10'):
        rank_str = '10'
        suit_str = input_str[2:]
    else:
        rank_str = input_str[0]
        suit_str = input_str[1:]

    # Map rank to index
    rank_map = {'3': 0, '4': 1, '5': 2, '6': 3, '7': 4, '8': 5,
                '9': 6, '10': 7, 'j': 8, 'q': 9, 'k': 10, 'a': 11, '2': 12}

    # Map suit to index
    suit_map = {'d': 0, '♦': 0, 'c': 1, '♣': 1, 'h': 2, '♥': 2, 's': 3, '♠': 3}

    if rank_str not in rank_map or suit_str not in suit_map:
        return None

    return rank_map[rank_str] * 4 + suit_map[suit_str]


def parse_move_input(input_str: str, hand: List[int]) -> Optional[List[int]]:
    """Parse user input into list of card indices."""
    input_str = input_str.strip().lower()

    # Check for pass
    if input_str in ['pass', 'p', '']:
        return []

    # Parse space-separated cards
    card_strs = input_str.split()
    cards = []

    for card_str in card_strs:
        card_idx = parse_card_input(card_str)
        if card_idx is None:
            print(f"Invalid card: {card_str}")
            return None
        if card_idx not in hand:
            print(f"You don't have {card_to_str(card_idx)}")
            return None
        cards.append(card_idx)

    return sorted(cards)


def display_hand(hand: List[int]):
    """Display player's hand in readable format."""
    print("\nYour hand:")
    sorted_hand = sorted(hand)

    # Group by rank for better readability
    by_rank = {}
    for card in sorted_hand:
        rank = card_rank(card)
        if rank not in by_rank:
            by_rank[rank] = []
        by_rank[rank].append(card)

    # Display each rank group
    for rank in sorted(by_rank.keys()):
        cards_str = ' '.join(card_to_str(c) for c in by_rank[rank])
        print(f"  {RANK_NAMES[rank]:3s}: {cards_str}")

    print(f"\nTotal cards: {len(hand)}")


def display_legal_moves(legal_moves: List[Move]):
    """Display numbered list of legal moves."""
    print("\nLegal moves:")

    # Always show pass if available
    for i, move in enumerate(legal_moves):
        if move.is_pass():
            print(f"  {i+1}. PASS")

    # Group moves by type
    singles = []
    pairs = []
    triples = []
    five_card = []

    for i, move in enumerate(legal_moves):
        if move.is_pass():
            continue

        move_type = detect_move_type(move.cards)
        if move_type == MoveType.SINGLE:
            singles.append((i, move))
        elif move_type == MoveType.PAIR:
            pairs.append((i, move))
        elif move_type == MoveType.TRIPLE:
            triples.append((i, move))
        else:
            five_card.append((i, move))

    # Display grouped
    if singles:
        print("  Singles:")
        for i, move in singles[:10]:  # Show first 10
            print(f"    {i+1}. {cards_to_str(move.cards)}")
        if len(singles) > 10:
            print(f"    ... and {len(singles)-10} more")

    if pairs:
        print("  Pairs:")
        for i, move in pairs:
            print(f"    {i+1}. {cards_to_str(move.cards)}")

    if triples:
        print("  Triples:")
        for i, move in triples:
            print(f"    {i+1}. {cards_to_str(move.cards)}")

    if five_card:
        print("  5-card hands:")
        for i, move in five_card:
            move_type = detect_move_type(move.cards)
            print(f"    {i+1}. {move_type.name}: {cards_to_str(move.cards)}")


def display_game_state(game: Big2Game, human_player: int):
    """Display current game state."""
    print("\n" + "=" * 60)
    print("Game State")
    print("=" * 60)

    # Show opponent hand sizes
    for i in range(4):
        if i == human_player:
            print(f"Player {i} (YOU): {len(game.hands[i])} cards")
        else:
            marker = " ←" if i == game.current_player else ""
            print(f"Player {i} (AI): {len(game.hands[i])} cards{marker}")

    # Show last move
    if game.last_move and not game.last_move.is_pass():
        print(f"\nLast move: Player {game.last_move.player} played {cards_to_str(game.last_move.cards)}")
        move_type = detect_move_type(game.last_move.cards)
        print(f"  Type: {move_type.name}")
    elif game.first_move:
        print("\nFirst move of the game (must contain 3♦)")
    else:
        print("\nStarting new trick (can play anything)")

    print("=" * 60)


def play_interactive_game(model_path: str, human_player: int = 0):
    """Play interactive game against AI."""
    # Load model
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = SimpleNetwork(**NETWORK_CONFIG).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    print(f"Loaded model from: {model_path}")
    print(f"Using device: {device}")
    print(f"You are Player {human_player}")

    # Initialize game
    game = Big2Game()

    # Game loop
    while not game.done:
        current = game.current_player

        if current == human_player:
            # Human turn
            display_game_state(game, human_player)
            display_hand(game.hands[human_player])

            legal_moves = get_legal_moves(game, human_player)

            # Get move from user
            while True:
                print("\nEnter your move:")
                print("  - Type card names (e.g., '3d 4d 5d')")
                print("  - Type 'pass' to pass")
                print("  - Type 'help' to see legal moves")
                print("  - Type 'quit' to exit")

                try:
                    user_input = input("> ").strip().lower()
                except EOFError:
                    print("\nGame quit.")
                    return

                if user_input == 'quit':
                    print("Game quit.")
                    return

                if user_input == 'help':
                    display_legal_moves(legal_moves)
                    continue

                # Parse move
                cards = parse_move_input(user_input, game.hands[human_player])
                if cards is None:
                    print("Invalid input. Try again.")
                    continue

                # Create move
                move = Move(cards, human_player)

                # Validate move
                valid = False
                for legal_move in legal_moves:
                    if sorted(legal_move.cards) == sorted(move.cards):
                        valid = True
                        move = legal_move
                        break

                if not valid:
                    print("That move is not legal. Type 'help' to see legal moves.")
                    continue

                # Execute move
                break

            # Display what human played
            if move.is_pass():
                print(f"\nYou passed.")
            else:
                print(f"\nYou played: {cards_to_str(move.cards)}")
                move_type = detect_move_type(move.cards)
                print(f"  Type: {move_type.name}")

        else:
            # AI turn
            legal_moves = get_legal_moves(game, current)
            move = select_action_greedy(game, current, model, device)

            # Display what AI played
            if move.is_pass():
                print(f"\nPlayer {current} (AI) passed.")
            else:
                print(f"\nPlayer {current} (AI) played: {cards_to_str(move.cards)}")
                move_type = detect_move_type(move.cards)
                print(f"  Type: {move_type.name}")

            try:
                input("Press Enter to continue...")
            except EOFError:
                print("\nGame quit.")
                return

        # Step game
        _, _, done, info = game.step(move)

    # Game over
    print("\n" + "=" * 60)
    print("GAME OVER!")
    print("=" * 60)
    winner = info["winner"]
    if winner == human_player:
        print("YOU WIN!")
    else:
        print(f"Player {winner} (AI) wins.")

    print(f"\nFinal scores: {info['all_rewards']}")
    print("=" * 60)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Play Big 2 against AI")

    parser.add_argument(
        "checkpoint",
        type=str,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--player",
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
        help="Which player position to play (0-3, default: 0)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible games"
    )

    args = parser.parse_args()

    # Set seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # Play game
    try:
        play_interactive_game(args.checkpoint, args.player)
    except KeyboardInterrupt:
        print("\nGame interrupted.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
