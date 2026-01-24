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


def score_remaining_hand(remaining: List[int]) -> Tuple[int, int]:
    """
    Score remaining hand after combo removal.

    From paper lines 39-220 - part of combo selection optimization.

    Algorithm:
    1. Build hand structure: triples → pairs → singles
    2. Find max card value across all groups
    3. Count number of groups

    Args:
        remaining: Cards left after removing combo(s)

    Returns:
        Tuple of (max_card_value, num_groups)
    """
    if not remaining:
        return (0, 0)

    # Build remaining hand structure
    hand_structure = []
    used = set()

    # Find triples first
    triples = get_triples(remaining)
    for triple in triples:
        hand_structure.append(triple)
        used.update(triple)

    # Find pairs (excluding cards in triples)
    remaining_after_triples = [c for c in remaining if c not in used]
    pairs = get_pairs(remaining_after_triples)
    for pair in pairs:
        hand_structure.append(pair)
        used.update(pair)

    # Remaining singles
    singles = [c for c in remaining if c not in used]
    for s in singles:
        hand_structure.append([s])

    # Find max card in any group
    max_card = max(max(group) for group in hand_structure)
    return (max_card, len(hand_structure))


def select_optimal_combos(hand: List[int]) -> List[List[int]]:
    """
    Select optimal combo(s) that minimize remaining hand strength.

    From paper lines 39-220 - exhaustive combo pairing optimization.

    Algorithm:
    1. Find all combos in hand
    2. Test single combos + all pairs of combos
    3. For each selection, score remaining hand
    4. Select combo(s) with lowest score:
       - Primary: lowest max remaining card
       - Secondary: fewest remaining groups
       - Tertiary: combo strength (250-point tiebreaker)

    Returns:
        Best combo or pair of combos (empty if no combos)
    """
    all_combos = get_combos(hand)

    if len(all_combos) == 0:
        return []
    if len(all_combos) == 1:
        return all_combos

    # Test all single and double combo combinations
    best_selection = []
    best_score = None  # (max_remaining, count, -combo_strength)

    # Test single combos
    for combo in all_combos:
        remaining = [c for c in hand if c not in combo]
        max_rem, count_rem = score_remaining_hand(remaining)
        combo_score = get_combo_score(combo)

        score = (max_rem, count_rem, -combo_score)

        if best_score is None or score < best_score:
            best_score = score
            best_selection = [combo]

    # Test pairs of combos
    for combo1, combo2 in combinations(all_combos, 2):
        # Check no overlap
        if set(combo1) & set(combo2):
            continue

        remaining = [c for c in hand if c not in combo1 and c not in combo2]
        max_rem, count_rem = score_remaining_hand(remaining)

        # Use max combo score for tiebreaking
        combo_score = max(get_combo_score(combo1), get_combo_score(combo2))

        score = (max_rem, count_rem, -combo_score)

        # Paper uses +250 advantage for tie-breaking (lines 119-121, 180-182)
        if best_score is None or score < best_score:
            best_score = score
            best_selection = [combo1, combo2]
        elif score[:2] == best_score[:2]:  # Tie on max_rem and count
            # Only switch if new combo is significantly stronger (+250)
            if -score[2] > -best_score[2] + 250:
                best_score = score
                best_selection = [combo1, combo2]

    return best_selection


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

    # Classify 5-card combos with dynamic thresholds
    # Use optimal combo selection (paper lines 39-220)
    combos = select_optimal_combos(hand)
    if combos:
        # Get all possible combos from unseen cards
        unseen_combos = get_combos(unseen)  # Full exhaustive search for maximum accuracy

        if unseen_combos:
            # Calculate dynamic percentile thresholds (paper lines 257-274)
            unseen_scores = sorted([get_combo_score(c) for c in unseen_combos])

            max_combo = max(unseen_scores)
            min_combo = min(unseen_scores)
            idx_70 = int(len(unseen_scores) * 0.7)
            div_combo = unseen_scores[idx_70] if idx_70 < len(unseen_scores) else max_combo

            # Classify based on dynamic thresholds
            for combo in combos:
                score = get_combo_score(combo)
                if score > max_combo:
                    classA.append(combo)
                elif score > div_combo:
                    classB.append(combo)
                elif score > min_combo:
                    classC.append(combo)
                else:
                    classD.append(combo)
        else:
            # No unseen combos - all our combos are class A
            for combo in combos:
                classA.append(combo)

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
# Split Move Logic
# =============================================================================

def split_combo_to_pairs(combo: List[int]) -> List[List[int]]:
    """
    Extract pairs from a combo (only works for full house and four-of-a-kind).

    From paper lines 557-575.

    Args:
        combo: 5-card combo

    Returns:
        List of pairs extracted from combo (empty for straight/flush/SF)
    """
    if not _is_full_house(combo) and not _is_four_of_kind(combo):
        return []

    pairs = []

    if _is_four_of_kind(combo):
        # Find the four cards of same rank
        ranks = [card_rank(c) for c in combo]
        rank_counts = Counter(ranks)
        four_rank = [r for r, cnt in rank_counts.items() if cnt == 4][0]
        four_cards = sorted([c for c in combo if card_rank(c) == four_rank])

        # Split into 2 pairs
        pairs.append(four_cards[0:2])
        pairs.append(four_cards[2:4])

    elif _is_full_house(combo):
        # Find the triple
        ranks = [card_rank(c) for c in combo]
        rank_counts = Counter(ranks)
        triple_rank = [r for r, cnt in rank_counts.items() if cnt == 3][0]
        pair_rank = [r for r, cnt in rank_counts.items() if cnt == 2][0]

        triple_cards = sorted([c for c in combo if card_rank(c) == triple_rank])
        pair_cards = sorted([c for c in combo if card_rank(c) == pair_rank])

        # Extract 1 pair from triple + the existing pair = 2 pairs
        pairs.append(pair_cards)  # The existing pair
        pairs.append(triple_cards[0:2])  # First 2 from triple

    # Sort pairs by score
    pairs.sort(key=get_pair_score)
    return pairs


def split_combo_to_singles(combo: List[int]) -> List[List[int]]:
    """
    Extract singles from a combo.

    From paper lines 615-616.

    Args:
        combo: 5-card combo

    Returns:
        List of single-card plays
    """
    return [[c] for c in combo]


def split_pair_to_singles(pair: List[int]) -> List[List[int]]:
    """
    Split a pair into two singles.

    From paper line 632.

    Args:
        pair: 2-card pair

    Returns:
        List of 2 single-card plays
    """
    return [[pair[0]], [pair[1]]]


def split_move(field: List[int], legal_card_lists: List[List[int]],
               combos: List[List[int]], pairs: List[List[int]],
               singles: List[List[int]], enemy_sizes: List[int]) -> List[int]:
    """
    Try to respond by splitting combinations when can't play full combos/pairs.

    From paper lines 634-700.

    Algorithm:
    - If field is combo: Try to play combo directly
    - If field is pair: Try pairs, else split combo to pairs
    - If field is single: Try singles, else split pair, else split combo

    Args:
        field: Last non-pass move cards
        legal_card_lists: All legal moves as card lists
        combos: Available combos
        pairs: Available pairs
        singles: Available singles
        enemy_sizes: List of enemy hand sizes

    Returns:
        Selected cards to play (empty list = pass)
    """
    enemy_min = min(enemy_sizes)

    # Field is combo (5 cards)
    if len(field) == 5:
        if combos:
            for combo in combos:
                if combo in legal_card_lists:
                    return combo
        return []

    # Field is pair (2 cards)
    if len(field) == 2:
        if pairs:
            # If enemy has 2 cards, play highest pair to block
            if enemy_min == 2:
                for pair in reversed(pairs):
                    if pair in legal_card_lists:
                        return pair
            else:
                for pair in pairs:
                    if pair in legal_card_lists:
                        return pair

        # No pairs available - try splitting combo to pairs
        if combos:
            # Find combo with highest pair potential
            for combo in reversed(combos):
                if _is_full_house(combo) or _is_four_of_kind(combo):
                    split_pairs = split_combo_to_pairs(combo)
                    # Try pairs from highest to lowest if enemy has 2 cards
                    if enemy_min == 2:
                        for pair in reversed(split_pairs):
                            if pair in legal_card_lists:
                                return pair
                    else:
                        for pair in split_pairs:
                            if pair in legal_card_lists:
                                return pair
        return []

    # Field is single (1 card)
    if len(field) == 1:
        if singles:
            # If enemy has 1 card, play highest single
            if enemy_min == 1:
                for single in reversed(singles):
                    if single in legal_card_lists:
                        return single
            else:
                for single in singles:
                    if single in legal_card_lists:
                        return single

        # No singles available - try splitting pair
        if pairs:
            highest_pair = pairs[-1]
            split_singles = split_pair_to_singles(highest_pair)
            for single in reversed(split_singles):
                if single in legal_card_lists:
                    return single

        # No pairs - try splitting combo to singles
        if combos:
            highest_combo = combos[-1]
            split_singles = split_combo_to_singles(highest_combo)
            # Play highest single if enemy has 1 card
            if enemy_min == 1:
                for single in reversed(split_singles):
                    if single in legal_card_lists:
                        return single
            else:
                for single in split_singles:
                    if single in legal_card_lists:
                        return single

        return []

    return []


# =============================================================================
# Advance Strategy
# =============================================================================

def calculate_move_len(classes: Dict) -> int:
    """
    Count total playable groups across all classes.

    From paper - used to determine if advance strategy should trigger.

    Args:
        classes: Card classification dict

    Returns:
        Total number of groups (combos + pairs + singles) across all classes
    """
    return sum(len(class_list) for class_list in classes.values())


def not_control_move(field: List[int], legal_card_lists: List[List[int]],
                     combos: List[List[int]], pairs: List[List[int]],
                     singles: List[List[int]], enemy_sizes: List[int]) -> List[int]:
    """
    Non-control move selection without splitting (for advance strategy).

    From paper lines 702-728.

    Args:
        field: Last non-pass move cards
        legal_card_lists: All legal moves as card lists
        combos: Available combos (all classes, weakest to strongest)
        pairs: Available pairs (all classes, weakest to strongest)
        singles: Available singles (all classes, weakest to strongest)
        enemy_sizes: List of enemy hand sizes

    Returns:
        Selected cards to play (empty list = pass)
    """
    enemy_min = min(enemy_sizes)

    # Field is combo (5 cards)
    if len(field) == 5:
        if combos:
            for combo in combos:
                if combo in legal_card_lists:
                    return combo
        return []

    # Field is pair (2 cards)
    if len(field) == 2:
        if pairs:
            # If enemy has 2 cards, play highest pair to block
            if enemy_min == 2:
                for pair in reversed(pairs):
                    if pair in legal_card_lists:
                        return pair
            else:
                for pair in pairs:
                    if pair in legal_card_lists:
                        return pair
        return []

    # Field is single (1 card)
    if len(field) == 1:
        if singles:
            # Play highest single (aggressive blocking)
            for single in reversed(singles):
                if single in legal_card_lists:
                    return single
        return []

    return []


def advance_strategy(hand: List[int], field: List[int],
                     legal_card_lists: List[List[int]],
                     classes: Dict, enemy_sizes: List[int],
                     control: bool, move_len: int) -> List[int]:
    """
    Aggressive close-win strategy when moveLen ≤ 3 and hand > 4.

    From paper lines 730-793.

    Triggered when close to winning (few moves left). More willing to
    play Class A cards and split combos to block opponents.

    Args:
        hand: Current hand
        field: Last non-pass move cards
        legal_card_lists: All legal moves as card lists
        classes: Card classification dict
        enemy_sizes: List of enemy hand sizes
        control: Whether we have control
        move_len: Total number of playable groups

    Returns:
        Selected cards to play (empty list = pass)
    """
    classA = classes['A']
    classB = classes['B']
    classC = classes['C']
    classD = classes['D']

    enemy_min = min(enemy_sizes)

    # Build card type lists (weakest to strongest)
    combos_all = (find_combos_in_class(classD) + find_combos_in_class(classC) +
                  find_combos_in_class(classB) + find_combos_in_class(classA))
    pairs_all = (find_pairs_in_class(classD) + find_pairs_in_class(classC) +
                 find_pairs_in_class(classB) + find_pairs_in_class(classA))
    singles_all = (find_singles(classD) + find_singles(classC) +
                   find_singles(classB) + find_singles(classA))

    # moveLen ≤ 2: Very close to winning
    if move_len <= 2:
        if control:
            # Play Class A first if available
            if len(classA) > 0:
                return classA[0]
            else:
                # Play weakest available
                if combos_all:
                    return combos_all[0]
                if pairs_all:
                    # If enemy has 2 cards, play highest pair
                    if enemy_min == 2:
                        return pairs_all[-1]
                    return pairs_all[0]
                if singles_all:
                    return singles_all[-1]  # Highest single
                return []

        else:  # Not control
            if enemy_min <= 2:
                # Aggressive blocking - willing to split
                return split_move(field, legal_card_lists, combos_all, pairs_all, singles_all, enemy_sizes)
            else:
                # Play without splitting
                return not_control_move(field, legal_card_lists, combos_all, pairs_all, singles_all, enemy_sizes)

    # moveLen == 3: Close to winning
    if move_len == 3:
        if control:
            # Play Class A only if have multiple Class A
            if len(classA) > 1:
                return classA[0]
            else:
                # Play weakest available
                if combos_all:
                    return combos_all[0]
                if pairs_all:
                    if enemy_min == 2:
                        return pairs_all[-1]
                    return pairs_all[0]
                if singles_all:
                    return singles_all[-1]  # Highest single
                return []

        else:  # Not control
            if enemy_min <= 2:
                # Aggressive blocking - willing to split
                return split_move(field, legal_card_lists, combos_all, pairs_all, singles_all, enemy_sizes)
            else:
                # Play without splitting
                return not_control_move(field, legal_card_lists, combos_all, pairs_all, singles_all, enemy_sizes)

    return []


# =============================================================================
# Holdback Logic
# =============================================================================

def check_holdback(hand: List[int], field: List[int],
                   enemy_sizes: List[int], turn: int,
                   pass_history: Dict[int, int],
                   classes: Dict, selected_cards: List[int]) -> bool:
    """
    Check if we should hold back (not play) the selected cards.

    From paper lines 469-501 - strategic card holding to prevent
    wasting high cards on low-value plays.

    Args:
        hand: Current hand
        field: Last non-pass move cards (empty if control)
        enemy_sizes: List of enemy hand sizes
        turn: Current turn number
        pass_history: Pass count for each player
        classes: Card classification dict
        selected_cards: Cards we're considering playing

    Returns:
        True if should HOLD (don't play), False if OK to play
    """
    classA = classes['A']
    classB = classes['B']
    classC = classes['C']
    classD = classes['D']

    hand_sorted = sorted(hand)
    enemy_min = min(enemy_sizes)

    # Holdback Single
    if len(field) == 1 or (not field and len(selected_cards) == 1):
        # Don't holdback if hand <= 2
        if len(hand) <= 2:
            return False

        # Don't holdback if enemy <= 2 and we're playing highest single (blocking)
        if enemy_min <= 2 and selected_cards == [hand_sorted[-1]]:
            return False

        # Holdback highest single if Class A is minority
        if len(classA) < (len(classB) + len(classC) + len(classD)):
            if selected_cards == [hand_sorted[-1]]:
                return True

        # Holdback highest single if all hands > 6
        if min([len(hand)] + enemy_sizes) > 6:
            if selected_cards == [hand_sorted[-1]]:
                return True

    # Holdback Pair
    if len(field) == 2 or (not field and len(selected_cards) == 2):
        # Don't holdback if hand <= 3
        if len(hand) <= 3:
            return False

        # Holdback strong pairs (score >= 50) if all hands > 2
        if min([len(hand)] + enemy_sizes) > 2:
            if get_pair_score(selected_cards) >= 50:
                return True

    # Holdback Combo
    if len(field) == 5 or (not field and len(selected_cards) == 5):
        # Don't holdback if hand == 5 (must play)
        if len(hand) == 5:
            return False

        # Holdback Class A/B combos in early game with deep stacks
        if min([len(hand)] + enemy_sizes) > 6 and turn <= 4:
            # Check if any enemy passed
            any_enemy_passed = any(count > 0 for count in pass_history.values())

            if any_enemy_passed:
                # Check if remaining hand has combos
                card_left = [c for c in hand if c not in selected_cards]
                remaining_combos = get_combos(card_left)

                # Check if selected is Class A or B
                is_class_a_or_b = False
                for combo in classA + classB:
                    if cards_match(combo, selected_cards):
                        is_class_a_or_b = True
                        break

                if len(remaining_combos) > 0 and is_class_a_or_b:
                    return True

    return False


# =============================================================================
# Non-Control Move Selection
# =============================================================================

def select_response_move(hand: List[int], field: List[int], classes: Dict,
                         legal_card_lists: List[List[int]],
                         enemy_sizes: List[int], turn: int,
                         pass_history: Dict[int, int]) -> List[int]:
    """
    Select a move when not in control (responding to opponent's play).

    Integrates holdback logic from paper lines 469-501.
    """
    classA = classes['A']
    classB = classes['B']
    classC = classes['C']
    classD = classes['D']

    enemy_min = min(enemy_sizes)

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
    # Use splitMove logic - willing to split combos/pairs
    if enemy_min == 1:
        combos_all = (find_combos_in_class(legal_D) + find_combos_in_class(legal_C) +
                      find_combos_in_class(legal_B) + find_combos_in_class(legal_A))
        pairs_all = (find_pairs_in_class(legal_D) + find_pairs_in_class(legal_C) +
                     find_pairs_in_class(legal_B) + find_pairs_in_class(legal_A))
        singles_all = (find_singles(legal_D) + find_singles(legal_C) +
                       find_singles(legal_B) + find_singles(legal_A))

        result = split_move(field, legal_card_lists, combos_all, pairs_all, singles_all, enemy_sizes)
        if result:
            return result
        # Fallback to pass if split_move returns empty
        return []

    # Normal play: try each class from weakest to strongest
    # Check holdback for each card before playing
    if legal_D:
        for selected_cards in legal_D:
            if not check_holdback(hand, field, enemy_sizes, turn, pass_history, classes, selected_cards):
                return selected_cards

    if legal_C:
        for selected_cards in legal_C:
            if not check_holdback(hand, field, enemy_sizes, turn, pass_history, classes, selected_cards):
                return selected_cards

    if legal_B:
        for selected_cards in legal_B:
            if not check_holdback(hand, field, enemy_sizes, turn, pass_history, classes, selected_cards):
                return selected_cards

    if legal_A:
        for selected_cards in legal_A:
            if not check_holdback(hand, field, enemy_sizes, turn, pass_history, classes, selected_cards):
                return selected_cards

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


def get_turn_number(game: Big2Game) -> int:
    """
    Get current turn number (0-indexed).

    A turn is one full round of all 4 players taking an action.
    From paper - used for holdback logic.
    """
    return len(game.move_history) // 4


def get_pass_history(game: Big2Game) -> Dict[int, int]:
    """
    Get pass count for each player in current trick.

    Counts consecutive passes since last non-pass move.
    From paper lines 469-501 - used for holdback logic.

    Returns:
        Dict mapping player index to number of consecutive passes
    """
    pass_count = {0: 0, 1: 0, 2: 0, 3: 0}

    # Look back until we find a non-pass move
    for move in reversed(game.move_history):
        if not move.is_pass():
            break
        pass_count[move.player] += 1

    return pass_count


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

    # Get turn number and pass history (for holdback/advance strategies)
    turn = get_turn_number(game)
    pass_history = get_pass_history(game)

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

    # Check for advance strategy (close to winning)
    # Trigger when moveLen ≤ 3 and hand > 4
    move_len = calculate_move_len(classes)
    if move_len <= 3 and len(hand) > 4:
        selected_cards = advance_strategy(hand, field, legal_card_lists, classes,
                                         enemy_sizes, control, move_len)
        if selected_cards:
            # Match selected cards to a legal move
            selected_set = set(selected_cards)
            for move in legal_moves:
                if not move.is_pass() and set(move.cards) == selected_set:
                    return move
            # Fallback if exact match not found
            for move in legal_moves:
                if not move.is_pass():
                    return move

    # Select cards to play
    if control:
        # We lead - use control strategies
        if len(hand) <= 4:
            selected_cards = under_four(hand, classA, classB, classC, classD, enemy_sizes)
        else:
            selected_cards = over_four(hand, classA, classB, classC, classD, enemy_sizes)
    else:
        # Responding - filter by legal moves with holdback logic
        selected_cards = select_response_move(hand, field, classes, legal_card_lists,
                                             enemy_sizes, turn, pass_history)

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
