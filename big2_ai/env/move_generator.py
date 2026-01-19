"""Legal move generation for Big 2."""

from typing import List, Set
from itertools import combinations
from collections import defaultdict

from .game import Move, card_rank, card_suit
from .move_detector import detect_move_type, can_beat, MoveType


def get_legal_moves(game, player: int) -> List[Move]:
    """
    Generate all legal moves for a player.

    Args:
        game: Big2Game instance
        player: Player index (0-3)

    Returns:
        List of legal Move objects
    """
    hand = game.hands[player]
    moves = []

    # First move must contain 3♦ (card 0)
    if game.first_move:
        moves.extend(generate_first_moves(hand, player))
        return moves

    # If there's a last move, can pass
    if game.last_move is not None:
        moves.append(Move([], player))

    # If no last move or starting new trick, can play anything
    if game.last_move is None:
        moves.extend(generate_all_moves(hand, player))
    else:
        # Must beat last move
        last_cards = game.last_move.cards
        moves.extend(generate_beating_moves(hand, last_cards, player))

    return moves


def generate_first_moves(hand: List[int], player: int) -> List[Move]:
    """
    Generate all legal first moves (must contain 3♦).

    For simplicity in Stage 1, we'll allow:
    - Single 3♦
    - Pairs containing 3♦
    - Triples containing 3♦
    - 5-card combos containing 3♦
    """
    moves = []

    if 0 not in hand:
        return moves

    # Single 3♦
    moves.append(Move([0], player))

    # Group cards by rank
    cards_by_rank = defaultdict(list)
    for card in hand:
        cards_by_rank[card_rank(card)].append(card)

    # Pairs with 3
    if len(cards_by_rank[0]) >= 2:
        for combo in combinations(cards_by_rank[0], 2):
            moves.append(Move(list(combo), player))

    # Triples with 3
    if len(cards_by_rank[0]) >= 3:
        for combo in combinations(cards_by_rank[0], 3):
            moves.append(Move(list(combo), player))

    # 5-card combos containing 3♦ (straights, flushes, etc.)
    for combo in combinations(hand, 5):
        if 0 in combo:
            move_type = detect_move_type(list(combo))
            if move_type in [MoveType.STRAIGHT, MoveType.FLUSH, MoveType.FULL_HOUSE,
                           MoveType.QUAD_WITH_KICKER, MoveType.STRAIGHT_FLUSH]:
                moves.append(Move(list(combo), player))

    return moves


def generate_all_moves(hand: List[int], player: int) -> List[Move]:
    """
    Generate all possible moves (when starting a new trick).
    """
    moves = []

    # Singles
    for card in hand:
        moves.append(Move([card], player))

    # Group cards by rank
    cards_by_rank = defaultdict(list)
    for card in hand:
        cards_by_rank[card_rank(card)].append(card)

    # Pairs
    for rank, cards in cards_by_rank.items():
        if len(cards) >= 2:
            for combo in combinations(cards, 2):
                moves.append(Move(list(combo), player))

    # Triples
    for rank, cards in cards_by_rank.items():
        if len(cards) >= 3:
            for combo in combinations(cards, 3):
                moves.append(Move(list(combo), player))

    # 5-card combinations
    if len(hand) >= 5:
        # Generate all 5-card combos and check if valid
        for combo in combinations(hand, 5):
            move_type = detect_move_type(list(combo))
            if move_type in [MoveType.STRAIGHT, MoveType.FLUSH, MoveType.FULL_HOUSE,
                           MoveType.QUAD_WITH_KICKER, MoveType.STRAIGHT_FLUSH]:
                moves.append(Move(list(combo), player))

    return moves


def generate_beating_moves(hand: List[int], last_cards: List[int], player: int) -> List[Move]:
    """
    Generate all moves that beat the last move.
    """
    moves = []
    last_type = detect_move_type(last_cards)
    last_len = len(last_cards)

    if last_len == 1:
        # Must play a higher single
        for card in hand:
            if can_beat([card], last_cards):
                moves.append(Move([card], player))

    elif last_len == 2:
        # Must play a higher pair
        cards_by_rank = defaultdict(list)
        for card in hand:
            cards_by_rank[card_rank(card)].append(card)

        for rank, cards in cards_by_rank.items():
            if len(cards) >= 2:
                for combo in combinations(cards, 2):
                    if can_beat(list(combo), last_cards):
                        moves.append(Move(list(combo), player))

    elif last_len == 3:
        # Must play a higher triple
        cards_by_rank = defaultdict(list)
        for card in hand:
            cards_by_rank[card_rank(card)].append(card)

        for rank, cards in cards_by_rank.items():
            if len(cards) >= 3:
                for combo in combinations(cards, 3):
                    if can_beat(list(combo), last_cards):
                        moves.append(Move(list(combo), player))

    elif last_len == 5:
        # Must play a higher 5-card combo
        if len(hand) >= 5:
            for combo in combinations(hand, 5):
                if can_beat(list(combo), last_cards):
                    moves.append(Move(list(combo), player))

    return moves


def test_move_generator():
    """Test move generation."""
    from .game import Big2Game

    print("Testing move generator...")

    game = Big2Game(seed=42)
    print(f"Initial game state:")
    print(game)
    print()

    # Get first moves
    moves = get_legal_moves(game, game.current_player)
    print(f"Player {game.current_player} has {len(moves)} legal first moves")
    print(f"First 5 moves: {moves[:5]}")
    print()

    # Play first move
    game.step(moves[0])
    print(f"After first move: {game}")
    print()

    # Get next moves
    moves = get_legal_moves(game, game.current_player)
    print(f"Player {game.current_player} has {len(moves)} legal moves")
    print(f"First 5 moves: {moves[:5]}")

    print("\nTest complete!")


if __name__ == "__main__":
    test_move_generator()
