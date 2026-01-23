"""State and action encoding for Big 2."""

import numpy as np
from typing import List, Optional

from .game import Big2Game, Move, card_rank


# State dimensions (Stage 2 - Legitimate state encoding, no hidden information)
STATE_DIM = 167
# - hand: 52 dims (one-hot: cards in hand)
# - graveyard: 52 dims (one-hot: cards played to discard pile)
# - last_move: 52 dims (one-hot: cards in last non-pass move)
# - control_player: 4 dims (one-hot: who controls the trick, relative to self)
# - consecutive_passes: 1 dim (normalized: passes since last play / 3.0)
# - opponent_hand_sizes: 3 dims (how many cards each opponent has, normalized)
# - high_cards_in_graveyard: 3 dims (count of 2s/Aces/Kings in graveyard, normalized)

# Perfect state dimensions (for PTIE critic - sees all hands)
PERFECT_STATE_DIM = 321
# - all_hands: 208 dims (4 players × 52 cards one-hot)
# - graveyard: 52 dims (one-hot: cards played to discard pile)
# - last_move: 52 dims (one-hot: cards in last non-pass move)
# - control_player: 4 dims (one-hot: who controls the trick, absolute)
# - consecutive_passes: 1 dim (normalized: passes since last play / 3.0)
# - current_player: 4 dims (one-hot: whose turn it is)
# Total: 208 + 52 + 52 + 4 + 1 + 4 = 321
# Note: This is used ONLY during training for the critic network

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


def encode_graveyard(game: Big2Game) -> np.ndarray:
    """
    Encode cards that have been played to the discard pile.

    This is LEGITIMATE observable information - all players can see
    what cards have been played throughout the game.

    Args:
        game: Game state

    Returns:
        52-dim one-hot vector (1 = card has been played to discard)
    """
    graveyard = np.zeros(52, dtype=np.float32)

    # Iterate through move history and mark all played cards
    for move in game.move_history:
        if not move.is_pass():
            for card in move.cards:
                graveyard[card] = 1.0

    return graveyard


def encode_control_state(game: Big2Game, player: int) -> np.ndarray:
    """
    Encode who controls the trick and pass count.

    This is CRITICAL for strategy:
    - If you control the trick (last_move is None or you played it), you can play anything
    - The pass count tells you how close the trick is to ending

    Args:
        game: Game state
        player: Current player index

    Returns:
        5-dim vector:
        - control_player (4 dims): one-hot of who controls, relative to self
          [0] = self controls, [1-3] = opponent 1/2/3 controls
        - consecutive_passes (1 dim): normalized (0.0 to 1.0)
    """
    control_state = np.zeros(5, dtype=np.float32)

    # Determine who controls the trick
    if game.last_move is None:
        # No last move = current player controls (can play anything)
        # But this happens AFTER 3 passes, so the player who played last controls
        # Actually at game start or after 3 passes, current player has control
        control_player_abs = player  # Current player has control
    else:
        # The player who made the last non-pass move controls
        control_player_abs = game.last_move.player

    # Convert to relative position (0 = self, 1-3 = opponents in order)
    relative_control = (control_player_abs - player) % 4
    control_state[relative_control] = 1.0

    # Consecutive passes (normalized)
    control_state[4] = game.passes_since_last_move / 3.0

    return control_state


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


def encode_high_cards_in_graveyard(game: Big2Game) -> np.ndarray:
    """
    Encode count of high cards (2s, Aces, Kings) that are in the graveyard.

    This is LEGITIMATE observable information - we only count cards
    that have been played (visible to all players).

    This helps the model understand:
    - How many high cards have been used up
    - Whether remaining high cards are still in play

    The model can compute: unknown = 4 - in_hand - in_graveyard

    In Big 2:
    - 2 is highest rank (rank 12)
    - Ace is second (rank 11)
    - King is third (rank 10)

    Args:
        game: Game state

    Returns:
        3-dim vector: [2s_in_graveyard, aces_in_graveyard, kings_in_graveyard]
        Normalized by 4 (max possible of each rank)
    """
    high_cards = np.zeros(3, dtype=np.float32)

    # Count high cards in the graveyard (played cards)
    for move in game.move_history:
        if not move.is_pass():
            for card in move.cards:
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


# Keep old function for backwards compatibility but mark as deprecated
def encode_remaining_high_cards(game: Big2Game, player: int) -> np.ndarray:
    """
    DEPRECATED: Use encode_high_cards_in_graveyard() instead.

    This function counts high cards "not in hand" which is ambiguous
    (includes both graveyard and opponent hands).
    """
    import warnings
    warnings.warn(
        "encode_remaining_high_cards is deprecated, use encode_high_cards_in_graveyard",
        DeprecationWarning
    )
    high_cards = np.zeros(3, dtype=np.float32)

    for card in range(52):
        if card in game.hands[player]:
            continue

        rank = card_rank(card)
        if rank == 12:
            high_cards[0] += 1.0
        elif rank == 11:
            high_cards[1] += 1.0
        elif rank == 10:
            high_cards[2] += 1.0

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
    Encode game state for player (Stage 2 - Legitimate state encoding).

    All information is OBSERVABLE by the player - no hidden information.

    State structure (167 dims):
    - hand (52): one-hot encoding of cards in hand
    - graveyard (52): one-hot encoding of cards played to discard
    - last_move (52): one-hot encoding of cards in last move
    - control_state (5): who controls trick (4) + consecutive passes (1)
    - opponent_hand_sizes (3): how many cards each opponent has (normalized)
    - high_cards_in_graveyard (3): count of 2s/Aces/Kings in discard (normalized)

    Args:
        game: Big2Game instance
        player: Player index (0-3)

    Returns:
        State vector of shape (167,)
    """
    hand_enc = encode_hand(game, player)                           # 52
    graveyard_enc = encode_graveyard(game)                         # 52
    last_move_enc = encode_last_move(game)                         # 52
    control_state_enc = encode_control_state(game, player)         # 5
    opponent_hand_sizes_enc = encode_opponent_hand_sizes(game, player)  # 3
    high_cards_graveyard_enc = encode_high_cards_in_graveyard(game)     # 3

    # Stage 2 state: 167 dims (all legitimate, observable information)
    return np.concatenate([
        hand_enc,               # 52: cards in hand
        graveyard_enc,          # 52: cards played to discard (NEW)
        last_move_enc,          # 52: last non-pass move
        control_state_enc,      # 5: control player (4) + passes (1) (NEW)
        opponent_hand_sizes_enc,    # 3: opponent card counts
        high_cards_graveyard_enc    # 3: high cards in graveyard (FIXED)
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


def encode_perfect_state(game: Big2Game) -> np.ndarray:
    """
    Encode PERFECT game state for PTIE critic network.

    This encoding includes ALL hidden information (all 4 hands).
    It is used ONLY during training for the critic network.
    The critic learns to estimate values with perfect information,
    which helps stabilize training of the actor (which only sees
    observable information).

    Perfect state structure (269 dims):
    - all_hands (208): 4 players × 52 cards one-hot
    - graveyard (52): one-hot encoding of cards played to discard
    - last_move (52): one-hot encoding of cards in last move
    - control_player (4): one-hot of who controls the trick (absolute)
    - consecutive_passes (1): normalized passes since last play
    - current_player (4): one-hot of whose turn it is

    Args:
        game: Big2Game instance

    Returns:
        Perfect state vector of shape (269,)
    """
    # Encode all 4 hands (208 dims)
    all_hands = np.zeros(208, dtype=np.float32)
    for player in range(4):
        offset = player * 52
        for card in game.hands[player]:
            all_hands[offset + card] = 1.0

    # Graveyard (52 dims)
    graveyard = encode_graveyard(game)

    # Last move (52 dims)
    last_move = encode_last_move(game)

    # Control player - absolute position (4 dims)
    control_player = np.zeros(4, dtype=np.float32)
    if game.last_move is None:
        # No last move = current player controls
        control_player[game.current_player] = 1.0
    else:
        # The player who made the last non-pass move controls
        control_player[game.last_move.player] = 1.0

    # Consecutive passes (1 dim)
    consecutive_passes = np.array([game.passes_since_last_move / 3.0], dtype=np.float32)

    # Current player (4 dims)
    current_player = np.zeros(4, dtype=np.float32)
    current_player[game.current_player] = 1.0

    # Concatenate all features (269 dims total)
    return np.concatenate([
        all_hands,           # 208: all 4 hands
        graveyard,           # 52: cards in discard
        last_move,           # 52: last non-pass move
        control_player,      # 4: who controls trick
        consecutive_passes,  # 1: pass count
        current_player       # 4: whose turn
    ])


def test_encoding():
    """Test state and action encoding (Stage 2 - Legitimate state encoding)."""
    from .game import Big2Game, Move

    print("Testing state and action encoding (Stage 2 - Legitimate)...")
    print("=" * 60)

    game = Big2Game(seed=42)
    print(f"Game state:")
    print(game)
    print()

    # Encode state for player 0
    state = encode_state(game, 0)
    print(f"State shape: {state.shape}")
    print(f"Expected dimensions: {STATE_DIM}")
    assert state.shape[0] == STATE_DIM, f"State shape mismatch: {state.shape[0]} != {STATE_DIM}"

    # Verify state structure (167 dims total)
    print("\nState structure breakdown:")
    print(f"  Hand (0-51):           sum = {state[:52].sum():.0f} (should be 13 cards)")
    print(f"  Graveyard (52-103):    sum = {state[52:104].sum():.0f} (should be 0 at start)")
    print(f"  Last move (104-155):   sum = {state[104:156].sum():.0f} (should be 0 at start)")
    print(f"  Control state (156-160): {state[156:161]}")
    print(f"    Control player one-hot: {state[156:160]}")
    print(f"    Consecutive passes: {state[160]:.2f}")
    print(f"  Opponent hand sizes (161-163): {state[161:164]}")
    print(f"  High cards in graveyard (164-166): {state[164:167]}")
    print()

    # Test individual encoding functions
    print("Testing individual encoding functions:")
    print("-" * 40)

    # Test graveyard encoding (should be empty at start)
    graveyard = encode_graveyard(game)
    print(f"Graveyard shape: {graveyard.shape} (expected (52,))")
    print(f"Graveyard sum: {graveyard.sum():.0f} (should be 0 at start)")

    # Test control state
    control = encode_control_state(game, 0)
    print(f"Control state shape: {control.shape} (expected (5,))")
    print(f"Control state: {control}")
    print(f"  (Player 0 should have control at start)")

    # Test hand sizes
    hand_sizes = encode_opponent_hand_sizes(game, 0)
    print(f"Hand sizes shape: {hand_sizes.shape} (expected (3,))")
    print(f"Hand sizes (normalized): {hand_sizes} (should all be 1.0 = 13/13)")

    # Test high cards in graveyard
    high_cards = encode_high_cards_in_graveyard(game)
    print(f"High cards in graveyard shape: {high_cards.shape} (expected (3,))")
    print(f"High cards [2s, As, Ks] in graveyard: {high_cards} (should be 0s at start)")
    print()

    # Simulate some moves and test again
    print("After playing some moves:")
    print("-" * 40)

    # Find who has the 3 of diamonds (card 0)
    starting_player = game.current_player
    print(f"Starting player (has 3d): {starting_player}")

    # Play a move (3 of diamonds must be first)
    first_move = Move([0], starting_player)  # 3d (card 0)
    game.step(first_move)

    # Next player plays
    next_player = game.current_player
    # Find a card they can play (any single card higher than 3d)
    next_hand = game.hands[next_player]
    playable_card = next_hand[-1]  # Highest card in hand
    second_move = Move([playable_card], next_player)
    game.step(second_move)

    # Test graveyard now has cards
    graveyard = encode_graveyard(game)
    print(f"Graveyard sum after 2 moves: {graveyard.sum():.0f} (should be 2)")
    assert graveyard.sum() == 2, "Graveyard should have 2 cards"

    # Test control state (second player should have control)
    current = game.current_player
    control = encode_control_state(game, current)
    print(f"Control state (player {current}'s view): {control}")
    print(f"  Control should be with previous player")

    # Test after passes
    pass_move = Move([], current)
    game.step(pass_move)
    next_current = game.current_player
    control = encode_control_state(game, next_current)
    print(f"Control after 1 pass: passes = {control[4]:.2f} (should be ~0.33)")
    print()

    # Test move history
    move_history = encode_move_history(game, max_moves=16)
    print(f"Move history shape: {move_history.shape}")
    print(f"Expected shape: ({HISTORY_LENGTH}, 52)")
    print(f"Move history sum: {move_history.sum():.2f} (should be 2 for 2 non-pass moves)")
    print()

    # Test action encoding
    print("Testing action encoding:")
    print("-" * 40)
    move = Move([0, 1, 2], 0)  # Triple 3s
    action = encode_action(move)
    print(f"Action shape: {action.shape}")
    print(f"Action dimensions: {ACTION_DIM}")
    print(f"Cards in action: {action.sum():.0f}")

    # Test pass
    pass_move = Move([], 0)
    pass_action = encode_action(pass_move)
    print(f"Pass action sum: {pass_action.sum():.0f} (should be 0)")
    print()

    # Test perfect state encoding (for PTIE)
    print("Testing PTIE perfect state encoding:")
    print("-" * 40)

    # Create fresh game for perfect state test
    game2 = Big2Game(seed=123)
    perfect_state = encode_perfect_state(game2)
    print(f"Perfect state shape: {perfect_state.shape}")
    print(f"Expected dimensions: {PERFECT_STATE_DIM}")
    assert perfect_state.shape[0] == PERFECT_STATE_DIM, f"Perfect state shape mismatch: {perfect_state.shape[0]} != {PERFECT_STATE_DIM}"

    # Verify structure (321 dims total)
    # all_hands: 0-207 (208), graveyard: 208-259 (52), last_move: 260-311 (52)
    # control: 312-315 (4), passes: 316 (1), current: 317-320 (4)
    print("\nPerfect state structure breakdown:")
    print(f"  All hands (0-207):       sum = {perfect_state[:208].sum():.0f} (should be 52 cards)")
    print(f"  Graveyard (208-259):     sum = {perfect_state[208:260].sum():.0f} (should be 0 at start)")
    print(f"  Last move (260-311):     sum = {perfect_state[260:312].sum():.0f} (should be 0 at start)")
    print(f"  Control player (312-315): {perfect_state[312:316]}")
    print(f"  Consecutive passes (316): {perfect_state[316]:.2f}")
    print(f"  Current player (317-320): {perfect_state[317:321]}")

    # Verify hand counts
    for p in range(4):
        hand_sum = perfect_state[p*52:(p+1)*52].sum()
        print(f"  Player {p} hand: {hand_sum:.0f} cards")
        assert hand_sum == 13, f"Player {p} should have 13 cards"

    print()
    print("=" * 60)
    print("All encoding tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_encoding()
