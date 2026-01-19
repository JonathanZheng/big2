"""State and action encoding for Big 2."""

import numpy as np
from typing import List, Optional

from .game import Big2Game, Move, card_rank


# State dimensions (simplified version for Stage 1)
STATE_DIM = 143
# - hand: 52 dims (one-hot: cards in hand)
# - last_move: 52 dims (one-hot: cards in last non-pass move)
# - opponent_counts: 39 dims (3 opponents × 13 card counts)

# Action dimensions
ACTION_DIM = 52
# - action: 52 dims (one-hot: cards in the action)


def encode_state(game: Big2Game, player: int) -> np.ndarray:
    """
    Encode game state from perspective of given player.

    State structure (143 dims):
    - hand (52): one-hot encoding of cards in hand
    - last_move (52): one-hot encoding of cards in last move
    - opponent_counts (39): 3 opponents × 13 card counts

    Args:
        game: Big2Game instance
        player: Player index (0-3)

    Returns:
        State vector of shape (143,)
    """
    state = np.zeros(STATE_DIM, dtype=np.float32)

    # Hand encoding (0-51)
    hand = game.hands[player]
    for card in hand:
        state[card] = 1.0

    # Last move encoding (52-103)
    if game.last_move is not None and not game.last_move.is_pass():
        for card in game.last_move.cards:
            state[52 + card] = 1.0

    # Opponent card counts (104-142)
    # For each opponent, count how many cards they have of each rank
    offset = 104
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

        # Add to state
        state[offset:offset + 13] = rank_counts
        offset += 13

    return state


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
    """Test state and action encoding."""
    from .game import Big2Game, Move

    print("Testing state and action encoding...")

    game = Big2Game(seed=42)
    print(f"Game state:")
    print(game)
    print()

    # Encode state
    state = encode_state(game, 0)
    print(f"State shape: {state.shape}")
    print(f"State dimensions: {STATE_DIM}")
    print(f"Hand encoding (first 52): sum = {state[:52].sum()}")
    print(f"Last move encoding (52-103): sum = {state[52:104].sum()}")
    print(f"Opponent counts (104-142): sum = {state[104:].sum():.2f}")
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

    print("\nTest complete!")


if __name__ == "__main__":
    test_encoding()
