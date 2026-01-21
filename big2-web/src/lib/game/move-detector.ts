/**
 * Move type detection for Big 2 - TypeScript port of move_detector.py
 */

import { Card, MoveType } from './types';
import { cardRank, cardSuit } from './constants';

function countByRank(cards: Card[]): Map<number, number> {
  const counts = new Map<number, number>();
  for (const card of cards) {
    const rank = cardRank(card);
    counts.set(rank, (counts.get(rank) || 0) + 1);
  }
  return counts;
}

function isStraight(ranks: number[]): boolean {
  if (ranks.length !== 5) return false;

  const sorted = [...ranks].sort((a, b) => a - b);

  // Check normal straight (consecutive ranks)
  for (let i = 0; i < 4; i++) {
    if (sorted[i + 1] !== sorted[i] + 1) {
      return false;
    }
  }
  return true;
}

function isFlush(suits: number[]): boolean {
  return new Set(suits).size === 1;
}

export function detectMoveType(cards: Card[]): MoveType {
  if (cards.length === 0) {
    return MoveType.PASS;
  }

  if (cards.length === 1) {
    return MoveType.SINGLE;
  }

  const ranks = cards.map(c => cardRank(c));
  const suits = cards.map(c => cardSuit(c));
  const rankCounts = countByRank(cards);

  if (cards.length === 2) {
    // Must be a pair
    if (rankCounts.size === 1) {
      return MoveType.PAIR;
    }
    return MoveType.INVALID;
  }

  if (cards.length === 3) {
    // Must be a triple
    if (rankCounts.size === 1) {
      return MoveType.TRIPLE;
    }
    return MoveType.INVALID;
  }

  if (cards.length === 5) {
    const isStr = isStraight(ranks);
    const isFl = isFlush(suits);

    if (isStr && isFl) {
      return MoveType.STRAIGHT_FLUSH;
    } else if (isStr) {
      return MoveType.STRAIGHT;
    } else if (isFl) {
      return MoveType.FLUSH;
    }

    // Check for full house (3+2) or quad with kicker (4+1)
    if (rankCounts.size === 2) {
      const counts = [...rankCounts.values()].sort((a, b) => a - b);

      if (counts[0] === 2 && counts[1] === 3) {
        return MoveType.FULL_HOUSE;
      }
      if (counts[0] === 1 && counts[1] === 4) {
        return MoveType.QUAD_WITH_KICKER;
      }
    }

    return MoveType.INVALID;
  }

  return MoveType.INVALID;
}

export function getMoveValue(cards: Card[], moveType: MoveType): [number, number] {
  if (moveType === MoveType.PASS) {
    return [-1, -1];
  }

  const ranks = cards.map(c => cardRank(c));
  const suits = cards.map(c => cardSuit(c));

  if (moveType === MoveType.SINGLE) {
    return [ranks[0], suits[0]];
  }

  if (moveType === MoveType.PAIR || moveType === MoveType.TRIPLE) {
    // Use the rank and highest suit
    return [ranks[0], Math.max(...suits)];
  }

  if (moveType === MoveType.STRAIGHT) {
    // Highest card in straight
    const maxRank = Math.max(...ranks);
    const maxCard = Math.max(...cards);
    return [maxRank, cardSuit(maxCard)];
  }

  if (moveType === MoveType.FLUSH) {
    // Compare by highest card
    const maxCard = Math.max(...cards);
    return [cardRank(maxCard), cardSuit(maxCard)];
  }

  if (moveType === MoveType.FULL_HOUSE) {
    // Compare by the triple
    const rankCounts = countByRank(cards);
    let tripleRank = -1;

    for (const [rank, count] of rankCounts) {
      if (count === 3) {
        tripleRank = rank;
        break;
      }
    }

    const tripleCards = cards.filter(c => cardRank(c) === tripleRank);
    const maxSuit = Math.max(...tripleCards.map(c => cardSuit(c)));
    return [tripleRank, maxSuit];
  }

  if (moveType === MoveType.QUAD_WITH_KICKER) {
    // Compare by the quad
    const rankCounts = countByRank(cards);
    let quadRank = -1;

    for (const [rank, count] of rankCounts) {
      if (count === 4) {
        quadRank = rank;
        break;
      }
    }

    const quadCards = cards.filter(c => cardRank(c) === quadRank);
    const maxSuit = Math.max(...quadCards.map(c => cardSuit(c)));
    return [quadRank, maxSuit];
  }

  if (moveType === MoveType.STRAIGHT_FLUSH) {
    // Highest card
    const maxCard = Math.max(...cards);
    return [cardRank(maxCard), cardSuit(maxCard)];
  }

  return [-1, -1];
}

export function canBeat(moveCards: Card[], lastMoveCards: Card[]): boolean {
  if (lastMoveCards.length === 0) {
    // No last move, any move is valid
    return true;
  }

  const moveType = detectMoveType(moveCards);
  const lastType = detectMoveType(lastMoveCards);

  if (moveType === MoveType.INVALID) {
    return false;
  }

  // Must play same number of cards
  if (moveCards.length !== lastMoveCards.length) {
    return false;
  }

  // For 5-card hands, special rules apply
  if (moveCards.length === 5) {
    // Rank of combo types (higher beats lower)
    const typeRanks: Record<MoveType, number> = {
      [MoveType.STRAIGHT]: 1,
      [MoveType.FLUSH]: 2,
      [MoveType.FULL_HOUSE]: 3,
      [MoveType.QUAD_WITH_KICKER]: 4,
      [MoveType.STRAIGHT_FLUSH]: 5,
      [MoveType.PASS]: 0,
      [MoveType.SINGLE]: 0,
      [MoveType.PAIR]: 0,
      [MoveType.TRIPLE]: 0,
      [MoveType.INVALID]: 0,
    };

    const moveRank = typeRanks[moveType];
    const lastRank = typeRanks[lastType];

    if (moveRank > lastRank) {
      return true;
    } else if (moveRank < lastRank) {
      return false;
    }
    // Same type, compare values (fall through)
  } else {
    // Must be same type
    if (moveType !== lastType) {
      return false;
    }
  }

  // Compare values
  const moveVal = getMoveValue(moveCards, moveType);
  const lastVal = getMoveValue(lastMoveCards, lastType);

  // Higher rank wins, or same rank but higher suit
  if (moveVal[0] > lastVal[0]) {
    return true;
  } else if (moveVal[0] === lastVal[0] && moveVal[1] > lastVal[1]) {
    return true;
  }

  return false;
}

// Utility for getting move type name
export function getMoveTypeName(moveType: MoveType): string {
  const names: Record<MoveType, string> = {
    [MoveType.PASS]: 'Pass',
    [MoveType.SINGLE]: 'Single',
    [MoveType.PAIR]: 'Pair',
    [MoveType.TRIPLE]: 'Triple',
    [MoveType.STRAIGHT]: 'Straight',
    [MoveType.FLUSH]: 'Flush',
    [MoveType.FULL_HOUSE]: 'Full House',
    [MoveType.QUAD_WITH_KICKER]: 'Four of a Kind',
    [MoveType.STRAIGHT_FLUSH]: 'Straight Flush',
    [MoveType.INVALID]: 'Invalid',
  };
  return names[moveType];
}
