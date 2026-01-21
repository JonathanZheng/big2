"""State and action encoding for Big 2."""

import numpy as np
from typing import List, Optional

from .game import Big2Game, Move, card_rank


# State dimensions (Stage 2 - MLP with enhanced features)
STATE_DIM = 149
# - hand: 52 dims (one-hot: cards in hand)
# - last_move: 52 dims (one-hot: cards in last non-pass move)
# - opponent_counts: 39 dims (3 opponents × 13 card counts)
# - opponent_hand_sizes: 3 dims (how many cards each opponent has, normalized)
# - remaining_high_cards: 3 dims (count of 2s/Aces/Kings not in hand, normalized)

# Action dimensions
ACTION_DIM = 52
# - action: 52 dims (one-hot: cards in the action)

# History dimensions (for LSTM)
HISTORY_LENGTH = 16


def encode_hand(game: Big2Game, player: int) -> np.ndarray:
    """
    Encode player's hand as one-hot vector.

    Args:
        game: Game state
        player: Player index

    Returns:
        52-dim one-hot vector (1 = card in hand)
    """
    hand_enc = np.zeros(52, dtype=np.float32)
    for card in game.hands[player]:
        hand_enc[card] = 1.0
    return hand_enc


def encode_last_move(game: Big2Game) -> np.ndarray:
    """
    Encode last non-pass move as one-hot vector.

    Args:
        game: Game state

    Returns:
        52-dim one-hot vector (1 = card in last move)
    """
    last_move_enc = np.zeros(52, dtype=np.float32)
    if game.last_move is not None and not game.last_move.is_pass():
        for card in game.last_move.cards:
            last_move_enc[card] = 1.0
    return last_move_enc


def encode_opponent_counts(game: Big2Game, player: int) -> np.ndarray:
    """
    Encode opponent card counts by rank.

    Args:
        game: Game state
        player: Current player index

    Returns:
        39-dim vector (3 opponents × 13 ranks, normalized by 4)
    """
    opponent_counts = []

    for i in range(4):
        if i == player:
            continue  # Skip self

        opponent_hand = game.hands[i]
        rank_counts = np.zeros(13, dtype=np.float32)

        for card in opponent_hand:
            rank = card_rank(card)
            rank_counts[rank] += 1.0

        # Normalize by max possible (4 cards per rank)
        rank_counts /= 4.0
        opponent_counts.append(rank_counts)

    return np.concatenate(opponent_counts)


def encode_opponent_hand_sizes(game: Big2Game, player: int) -> np.ndarray:
    """
    Encode opponent hand sizes (how many cards each opponent has).

    This helps the model understand:
    - Which opponents are close to winning (low card count)
    - Game progression (total cards remaining)

    Args:
        game: Game state
        player: Current player index

    Returns:
        3-dim vector (normalized by max hand size 13)
    """
    hand_sizes = np.zeros(3, dtype=np.float32)

    idx = 0
    for i in range(4):
        if i == player:
            continue  # Skip self
        hand_sizes[idx] = len(game.hands[i]) / 13.0  # Normalize
        idx += 1

    return hand_sizes


def encode_remaining_high_cards(game: Big2Game, player: int) -> np.ndarray:
    """
    Encode count of remaining high cards (2s, Aces, Kings) not in player's hand.

    This helps the model understand:
    - Hand strength (do I have the highest cards?)
    - Whether opponents likely have strong cards

    In Big 2:
    - 2 is highest rank (rank 12)
    - Ace is second (rank 11)
    - King is third (rank 10)

    Args:
        game: Game state
        player: Current player index

    Returns:
        3-dim vector: [remaining_2s, remaining_aces, remaining_kings]
    """
    high_cards = np.zeros(3, dtype=np.float32)

    # Count high cards NOT in current player's hand
    for card in range(52):
        if card in game.hands[player]:
            continue  # Skip cards in our hand

        rank = card_rank(card)
        if rank == 12:  # 2 (highest)
            high_cards[0] += 1.0
        elif rank == 11:  # Ace
            high_cards[1] += 1.0
        elif rank == 10:  # King
            high_cards[2] += 1.0

    # Normalize by max possible (4 of each rank)
    high_cards /= 4.0

    return high_cards


def encode_opponent_union(game: Big2Game, player: int) -> np.ndarray:
    """
    Encode union of all cards played/seen by opponents.

    Returns 52-dim one-hot vector where 1 = card has been played or is not
    in current player's hand (i.e., card is "seen" or eliminated from consideration).

    Args:
        game: Game state
        player: Current player index

    Returns:
        52-dim one-hot vector (1 = card seen/played)
    """
    opponent_union = np.zeros(52, dtype=np.float32)

    # Mark all cards NOT in current player's hand as "seen"
    for card in range(52):
        if card not in game.hands[player]:
            opponent_union[card] = 1.0

    return opponent_union


def encode_move_history(game: Big2Game, max_moves: int = 16) -> np.ndarray:
    """
    Encode last N moves as sequence for LSTM.

    Args:
        game: Game state
        max_moves: Maximum number of moves to encode (default: 16)

    Returns:
        Array of shape (max_moves, 52) where each row is one-hot encoding
        of a move. Padded with zeros for early game states. Most recent
        move is at the end (last row).
    """
    history = np.zeros((max_moves, 52), dtype=np.float32)

    # Get last max_moves from game.move_history
    recent_moves = game.move_history[-max_moves:] if len(game.move_history) > 0 else []

    # Encode each move (fill from the end, most recent last)
    offset = max_moves - len(recent_moves)
    for i, move in enumerate(recent_moves):
        if not move.is_pass():
            for card in move.cards:
                history[offset + i, card] = 1.0
        # Pass moves remain all zeros

    return history


def encode_state(game: Big2Game, player: int) -> np.ndarray:
    """
    Encode game state for player (Stage 2 - MLP with enhanced features).

    State structure (149 dims):
    - hand (52): one-hot encoding of cards in hand
    - last_move (52): one-hot encoding of cards in last move
    - opponent_counts (39): 3 opponents × 13 card counts (normalized)
    - opponent_hand_sizes (3): how many cards each opponent has (normalized)
    - remaining_high_cards (3): count of 2s/Aces/Kings not in hand (normalized)

    Args:
        game: Big2Game instance
        player: Player index (0-3)

    Returns:
        State vector of shape (149,)
    """
    hand_enc = encode_hand(game, player)                      # 52
    last_move_enc = encode_last_move(game)                    # 52
    opponent_counts_enc = encode_opponent_counts(game, player) # 39
    opponent_hand_sizes_enc = encode_opponent_hand_sizes(game, player)  # 3
    remaining_high_cards_enc = encode_remaining_high_cards(game, player)  # 3

    # Stage 2 state: 149 dims
    return np.concatenate([
        hand_enc,
        last_move_enc,
        opponent_counts_enc,
        opponent_hand_sizes_enc,
        remaining_high_cards_enc
    ])


def encode_action(move: Move) -> np.ndarray:
    """
    Encode an action (move) as a one-hot vector.

    Action structure (52 dims):
    - Cards in the move are set to 1.0
    - Pass is encoded as all zeros

    Args:
        move: Move object

    Returns:
        Action vector of shape (52,)
    """
    action = np.zeros(ACTION_DIM, dtype=np.float32)

    if not move.is_pass():
        for card in move.cards:
            action[card] = 1.0

    return action


def test_encoding():
    """Test state and action encoding (Stage 2 - MLP with enhanced features)."""
    from .game import Big2Game, Move

    print("Testing state and action encoding (Stage 2)...")

    game = Big2Game(seed=42)
    print(f"Game state:")
    print(game)
    print()

    # Encode state
    state = encode_state(game, 0)
    print(f"State shape: {state.shape}")
    print(f"Expected dimensions: {STATE_DIM}")
    assert state.shape[0] == STATE_DIM, f"State shape mismatch: {state.shape[0]} != {STATE_DIM}"
    print(f"Hand encoding (0-51): sum = {state[:52].sum()}")
    print(f"Last move encoding (52-103): sum = {state[52:104].sum()}")
    print(f"Opponent counts (104-142): sum = {state[104:143].sum():.2f}")
    print(f"Opponent hand sizes (143-145): {state[143:146]}")
    print(f"Remaining high cards (146-148): {state[146:149]}")
    print()

    # Test new encoding functions individually
    hand_sizes = encode_opponent_hand_sizes(game, 0)
    print(f"Hand sizes shape: {hand_sizes.shape} (expected (3,))")
    print(f"Hand sizes (normalized): {hand_sizes}")
    print()

    high_cards = encode_remaining_high_cards(game, 0)
    print(f"Remaining high cards shape: {high_cards.shape} (expected (3,))")
    print(f"Remaining [2s, Aces, Kings] (normalized): {high_cards}")
    print()

    # Test move history
    move_history = encode_move_history(game, max_moves=16)
    print(f"Move history shape: {move_history.shape}")
    print(f"Expected shape: ({HISTORY_LENGTH}, 52)")
    print(f"Move history sum: {move_history.sum():.2f}")
    print()

    # Encode action
    move = Move([0, 1, 2], 0)  # Triple 3s
    action = encode_action(move)
    print(f"Action shape: {action.shape}")
    print(f"Action dimensions: {ACTION_DIM}")
    print(f"Cards in action: {action.sum()}")
    print()

    # Test pass
    pass_move = Move([], 0)
    pass_action = encode_action(pass_move)
    print(f"Pass action sum: {pass_action.sum()}")

    print("\nAll tests passed!")


if __name__ == "__main__":
    test_encoding()
