"""Rule-based AI agent from MDPI 2021 paper.

Implements the algorithm from:
"A Rule-Based AI Method for an Agent Playing Big Two"
Applied Sciences 11(9):4206 (2021)

Uses native card representation (0-51 indices) throughout.
Card ordering: 3D(0) < 3C(1) < 3H(2) < 3S(3) < 4D(4) < ... < 2S(51)
"""

from typing import List, Dict, Tuple, Optional
from collections import Counter
from itertools import combinations

from ..env import Big2Game, get_legal_moves
from ..env.game import card_rank, card_suit
from ..env.move_detector import detect_move_type, MoveType


# =============================================================================
# Helper Functions
# =============================================================================

def get_pair_score(cards: List[int]) -> int:
    """Score a pair by its highest card."""
    return max(cards)


def get_combo_score(cards: List[int]) -> int:
    """
    Score a 5-card combo.
    Base scores: Straight=1000, Flush=2000, FullHouse=3000, Four=4000, SF=5000
    """
    is_s = _is_straight(cards)
    is_f = _is_flush(cards)
    is_fh = _is_full_house(cards)
    is_4 = _is_four_of_kind(cards)
    is_sf = is_s and is_f

    if is_sf:
        return 5000 + max(cards)
    if is_4:
        return 4000 + max(cards)
    if is_fh:
        # Score by the triple
        ranks = [card_rank(c) for c in cards]
        triple_rank = [r for r, cnt in Counter(ranks).items() if cnt == 3][0]
        return 3000 + triple_rank * 4 + 3
    if is_f:
        return 2000 + max(cards)
    if is_s:
        return 1000 + max(cards)
    return 0


def _is_straight(cards: List[int]) -> bool:
    """Check if 5 cards form a straight."""
    if len(cards) != 5:
        return False
    ranks = sorted([card_rank(c) for c in cards])
    for i in range(4):
        if ranks[i + 1] != ranks[i] + 1:
            return False
    return True


def _is_flush(cards: List[int]) -> bool:
    """Check if 5 cards are same suit."""
    if len(cards) != 5:
        return False
    return len(set(card_suit(c) for c in cards)) == 1


def _is_full_house(cards: List[int]) -> bool:
    """Check if 5 cards form full house (3+2)."""
    if len(cards) != 5:
        return False
    ranks = [card_rank(c) for c in cards]
    counts = sorted(Counter(ranks).values())
    return counts == [2, 3]


def _is_four_of_kind(cards: List[int]) -> bool:
    """Check if 5 cards have 4-of-a-kind."""
    if len(cards) != 5:
        return False
    ranks = [card_rank(c) for c in cards]
    counts = sorted(Counter(ranks).values())
    return counts == [1, 4]


# =============================================================================
# Card Finding Functions
# =============================================================================

def get_pairs(hand: List[int]) -> List[List[int]]:
    """Find all pairs in hand."""
    pairs = []
    hand_sorted = sorted(hand)
    i = 0
    while i < len(hand_sorted) - 1:
        if card_rank(hand_sorted[i]) == card_rank(hand_sorted[i + 1]):
            pairs.append([hand_sorted[i], hand_sorted[i + 1]])
            i += 2  # Skip both cards
        else:
            i += 1
    return pairs


def get_all_pairs(hand: List[int]) -> List[List[int]]:
    """Find all possible pairs (including overlapping) in hand."""
    pairs = []
    hand_sorted = sorted(hand)
    for i in range(len(hand_sorted) - 1):
        for j in range(i + 1, len(hand_sorted)):
            if card_rank(hand_sorted[i]) == card_rank(hand_sorted[j]):
                pairs.append([hand_sorted[i], hand_sorted[j]])
    return pairs


def get_triples(hand: List[int]) -> List[List[int]]:
    """Find all triples in hand."""
    triples = []
    ranks = [card_rank(c) for c in hand]
    rank_counts = Counter(ranks)
    for rank, count in rank_counts.items():
        if count >= 3:
            cards_of_rank = sorted([c for c in hand if card_rank(c) == rank])
            triples.append(cards_of_rank[:3])
    return triples


def get_combos(hand: List[int]) -> List[List[int]]:
    """Find all 5-card combos (straights, flushes, full houses, etc.)."""
    combos = []
    if len(hand) < 5:
        return combos

    for combo in combinations(hand, 5):
        combo_list = list(combo)
        if (_is_straight(combo_list) or _is_flush(combo_list) or
                _is_full_house(combo_list) or _is_four_of_kind(combo_list)):
            combos.append(combo_list)
    return combos


# =============================================================================
# Classification System
# =============================================================================

def enemy_probable_cards(hand: List[int], graveyard: List[int]) -> List[int]:
    """Get cards that opponents might have."""
    all_cards = set(range(52))
    seen = set(hand) | set(graveyard)
    return sorted(all_cards - seen)


def classify_cards(hand: List[int], graveyard: List[int]) -> Dict[str, List]:
    """
    Classify hand cards into A/B/C/D classes.

    Class A: Higher than any enemy could have (unbeatable)
    Class B: Top 30% of enemy range (strong)
    Class C: Middle 50% of enemy range (medium)
    Class D: Bottom 20% of enemy range (weak)
    """
    unseen = enemy_probable_cards(hand, graveyard)

    if not unseen:
        # All cards seen - everything is class A
        return {
            'A': [[c] for c in sorted(hand)],
            'B': [], 'C': [], 'D': []
        }

    # Compute thresholds for single card classification
    max_unseen = max(unseen)
    min_unseen = min(unseen)
    idx_80 = int(len(unseen) * 0.8)
    div_80 = unseen[idx_80] if idx_80 < len(unseen) else max_unseen

    classA, classB, classC, classD = [], [], [], []

    # Classify singles
    for card in sorted(hand):
        if card > max_unseen:
            classA.append([card])
        elif card > div_80:
            classB.append([card])
        elif card > min_unseen:
            classC.append([card])
        else:
            classD.append([card])

    # Classify pairs
    pairs = get_pairs(hand)
    unseen_pairs = get_all_pairs(unseen)

    if pairs:
        if unseen_pairs:
            pair_scores = [get_pair_score(p) for p in unseen_pairs]
            max_pair = max(pair_scores)
            min_pair = min(pair_scores)
            idx_70 = int(len(pair_scores) * 0.7)
            div_pair = pair_scores[idx_70] if idx_70 < len(pair_scores) else max_pair

            for pair in pairs:
                score = get_pair_score(pair)
                if score > max_pair:
                    classA.append(pair)
                elif score > div_pair:
                    classB.append(pair)
                elif score > min_pair:
                    classC.append(pair)
                else:
                    classD.append(pair)
        else:
            # No unseen pairs - all our pairs are class A
            for pair in pairs:
                classA.append(pair)

    # Classify 5-card combos
    combos = get_combos(hand)
    for combo in combos:
        score = get_combo_score(combo)
        # Thresholds based on paper's logic
        if score > 4000:  # Four of a kind or straight flush
            classA.append(combo)
        elif score > 3000:  # Full house
            classB.append(combo)
        elif score > 2000:  # Flush
            classC.append(combo)
        else:  # Straight
            classD.append(combo)

    return {'A': classA, 'B': classB, 'C': classC, 'D': classD}


# =============================================================================
# Move Selection Helpers
# =============================================================================

def find_singles(class_list: List[List[int]]) -> List[List[int]]:
    """Find single-card plays in class list."""
    return [c for c in class_list if len(c) == 1]


def find_pairs_in_class(class_list: List[List[int]]) -> List[List[int]]:
    """Find pair plays in class list."""
    return [c for c in class_list if len(c) == 2]


def find_combos_in_class(class_list: List[List[int]]) -> List[List[int]]:
    """Find 5-card combo plays in class list."""
    return [c for c in class_list if len(c) == 5]


def cards_match(cards1: List[int], cards2: List[int]) -> bool:
    """Check if two card lists are the same (order independent)."""
    return set(cards1) == set(cards2)


def find_matching_class(cards: List[int], classes: Dict) -> Optional[str]:
    """Find which class a card combination belongs to."""
    for class_name in ['A', 'B', 'C', 'D']:
        for class_cards in classes[class_name]:
            if cards_match(cards, class_cards):
                return class_name
    return None


# =============================================================================
# End-Game Strategies (4 or fewer cards)
# =============================================================================

def two_card_strategy(hand: List[int], classA: List, enemy_min: int) -> List[int]:
    """Strategy for 2 cards remaining."""
    hand_sorted = sorted(hand)
    pairs = get_pairs(hand)

    if pairs:
        return pairs[0]
    elif classA or enemy_min == 1:
        return [hand_sorted[1]]  # Higher card
    return [hand_sorted[0]]  # Lower card


def three_card_strategy(hand: List[int], classA: List, classB: List,
                        classC: List, classD: List, enemy_min: int) -> List[int]:
    """Strategy for 3 cards remaining."""
    hand_sorted = sorted(hand)
    pairs = get_pairs(hand)

    if pairs:
        pair = pairs[0]
        single = [c for c in hand if c not in pair][0]

        if classA:
            # Check if pair is in class A
            for a_cards in classA:
                if len(a_cards) == 2 and cards_match(a_cards, pair):
                    return pair
            # Check if single is in class A
            for a_cards in classA:
                if len(a_cards) == 1 and a_cards[0] == single:
                    return [single]

        if enemy_min == 1:
            return pair  # Block single with pair
        elif enemy_min == 2:
            return [single]  # Save pair for blocking pair

        # Default: play from weakest class
        if classD:
            return classD[0]
        if classC:
            return classC[0]
        if classB:
            return classB[0]
        if classA:
            return classA[0]

        return [single]
    else:
        # All singles
        if enemy_min == 1 and classA:
            return [hand_sorted[2]]  # Highest
        return [hand_sorted[0]]  # Lowest


def four_card_strategy(hand: List[int], classA: List, classB: List,
                       classC: List, classD: List, enemy_min: int) -> List[int]:
    """Strategy for 4 cards remaining."""
    hand_sorted = sorted(hand)
    pairs = get_pairs(hand)

    if len(pairs) == 2:
        # Two pairs - play based on enemy state
        pairs_sorted = sorted(pairs, key=get_pair_score)
        if enemy_min == 2:
            return pairs_sorted[1]  # Higher pair to block
        return pairs_sorted[0]  # Lower pair

    elif pairs:
        pair = pairs[0]
        singles = sorted([c for c in hand if c not in pair])

        if classA:
            # Check if high single is in class A
            for a_cards in classA:
                if len(a_cards) == 1 and a_cards[0] == singles[1]:
                    return [singles[0]]  # Play lower single

        if enemy_min == 1:
            return pair  # Block single
        elif enemy_min == 2:
            return [singles[0]]  # Save pair

        # Default: play lowest
        if classD:
            return classD[0]
        if classC:
            return classC[0]
        if classB:
            return classB[0]
        if classA:
            return classA[0]

        return [singles[0]]
    else:
        # All singles
        if classA:
            return [hand_sorted[1]]  # Second lowest
        elif enemy_min == 1:
            return [hand_sorted[3]]  # Highest
        return [hand_sorted[0]]  # Lowest


def under_four(hand: List[int], classA: List, classB: List,
               classC: List, classD: List, enemy_sizes: List[int]) -> List[int]:
    """End-game strategy for 4 or fewer cards."""
    enemy_min = min(enemy_sizes)

    if len(hand) == 1:
        return [hand[0]]
    elif len(hand) == 2:
        return two_card_strategy(hand, classA, enemy_min)
    elif len(hand) == 3:
        return three_card_strategy(hand, classA, classB, classC, classD, enemy_min)
    elif len(hand) == 4:
        return four_card_strategy(hand, classA, classB, classC, classD, enemy_min)

    return [sorted(hand)[0]]


# =============================================================================
# Mid-Game Strategy (more than 4 cards)
# =============================================================================

def over_four(hand: List[int], classA: List, classB: List,
              classC: List, classD: List, enemy_sizes: List[int]) -> List[int]:
    """Mid-game strategy for more than 4 cards - play weakest class first."""
    enemy_min = min(enemy_sizes)

    # Collect all plays by type, ordered weakest to strongest
    combos = (find_combos_in_class(classD) + find_combos_in_class(classC) +
              find_combos_in_class(classB) + find_combos_in_class(classA))
    pairs = (find_pairs_in_class(classD) + find_pairs_in_class(classC) +
             find_pairs_in_class(classB) + find_pairs_in_class(classA))
    singles = (find_singles(classD) + find_singles(classC) +
               find_singles(classB) + find_singles(classA))

    # If enemy is about to win, play multi-card hands aggressively
    if enemy_min == 1:
        if combos:
            return combos[0]
        if pairs:
            return pairs[0]

    # Normal play: prefer to shed cards evenly
    # If we have more pairs than singles, play a pair
    if len(pairs) > len(singles) and pairs:
        return pairs[0]

    # Default: play from weakest class
    if classD:
        return classD[0]
    if classC:
        return classC[0]
    if classB:
        return classB[0]
    if classA:
        return classA[0]

    # Fallback
    return [sorted(hand)[0]]


# =============================================================================
# Non-Control Move Selection
# =============================================================================

def select_response_move(field: List[int], classes: Dict,
                         legal_card_lists: List[List[int]],
                         enemy_min: int) -> List[int]:
    """Select a move when not in control (responding to opponent's play)."""
    classA = classes['A']
    classB = classes['B']
    classC = classes['C']
    classD = classes['D']

    # Filter each class to only legal moves
    def filter_legal(class_list):
        result = []
        for cards in class_list:
            for legal in legal_card_lists:
                if cards_match(cards, legal):
                    result.append(cards)
                    break
        return result

    legal_D = filter_legal(classD)
    legal_C = filter_legal(classC)
    legal_B = filter_legal(classB)
    legal_A = filter_legal(classA)

    # If enemy is close to winning, play more aggressively
    if enemy_min <= 2:
        # Play strongest available to try to regain control
        if legal_D:
            return legal_D[0]
        if legal_C:
            return legal_C[0]
        if legal_B:
            return legal_B[0]
        if legal_A:
            return legal_A[0]
        return []  # Pass

    # Normal play: play weakest available
    if legal_D:
        return legal_D[0]
    if legal_C:
        return legal_C[0]
    if legal_B:
        return legal_B[0]
    if legal_A:
        return legal_A[0]

    return []  # Pass


# =============================================================================
# Main Entry Point
# =============================================================================

def get_graveyard(game: Big2Game) -> List[int]:
    """Get all cards that have been played to the discard pile."""
    graveyard = []
    for move in game.move_history:
        if not move.is_pass():
            graveyard.extend(move.cards)
    return graveyard


def select_action_rule_based_bot(game: Big2Game, player: int):
    """
    Rule-based bot main entry point.
    Implements the MDPI 2021 paper algorithm using native card representation.

    Args:
        game: Current game state
        player: Player index (0-3)

    Returns:
        Selected Move object
    """
    legal_moves = get_legal_moves(game, player)

    if len(legal_moves) == 0:
        raise ValueError(f"No legal moves for player {player}")

    # If only one legal move, play it
    if len(legal_moves) == 1:
        return legal_moves[0]

    hand = game.hands[player]
    graveyard = get_graveyard(game)

    # Get enemy hand sizes
    enemy_sizes = [len(game.hands[i]) for i in range(4) if i != player]
    enemy_min = min(enemy_sizes)

    # Determine if we have control
    # Control = last non-pass move was ours, or no last move (game start / after 3 passes)
    if game.last_move is None:
        control = True  # Game start or after 3 passes - we can play anything
    else:
        control = (game.last_move.player == player)

    # Build field (last non-pass move cards)
    field = []
    if game.last_move and not game.last_move.is_pass():
        field = game.last_move.cards

    # Classify cards
    classes = classify_cards(hand, graveyard)
    classA = classes['A']
    classB = classes['B']
    classC = classes['C']
    classD = classes['D']

    # Convert legal moves to card lists for matching
    legal_card_lists = []
    for m in legal_moves:
        if m.is_pass():
            legal_card_lists.append([])
        else:
            legal_card_lists.append(list(m.cards))

    # Select cards to play
    if control:
        # We lead - use control strategies
        if len(hand) <= 4:
            selected_cards = under_four(hand, classA, classB, classC, classD, enemy_sizes)
        else:
            selected_cards = over_four(hand, classA, classB, classC, classD, enemy_sizes)
    else:
        # Responding - filter by legal moves
        selected_cards = select_response_move(field, classes, legal_card_lists, enemy_min)

    # Find matching move
    if not selected_cards:
        # Pass
        for move in legal_moves:
            if move.is_pass():
                return move
        # No pass available, play first move
        return legal_moves[0]

    # Match selected cards to a legal move
    selected_set = set(selected_cards)
    for move in legal_moves:
        if not move.is_pass() and set(move.cards) == selected_set:
            return move

    # Fallback: play first non-pass or pass
    for move in legal_moves:
        if not move.is_pass():
            return move
    return legal_moves[0]
