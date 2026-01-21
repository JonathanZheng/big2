/**
 * Legal move generation for Big 2 - TypeScript port of move_generator.py
 */

import { Card, Move, PlayerIndex, MoveType } from './types';
import { cardRank } from './constants';
import { Big2Game, createMove } from './game-engine';
import { detectMoveType, canBeat } from './move-detector';

// Helper: Generate all k-combinations of an array
function combinations<T>(arr: T[], k: number): T[][] {
  const result: T[][] = [];

  function helper(start: number, combo: T[]) {
    if (combo.length === k) {
      result.push([...combo]);
      return;
    }

    for (let i = start; i < arr.length; i++) {
      combo.push(arr[i]);
      helper(i + 1, combo);
      combo.pop();
    }
  }

  helper(0, []);
  return result;
}

// Group cards by rank
function groupByRank(cards: Card[]): Map<number, Card[]> {
  const groups = new Map<number, Card[]>();
  for (const card of cards) {
    const rank = cardRank(card);
    if (!groups.has(rank)) {
      groups.set(rank, []);
    }
    groups.get(rank)!.push(card);
  }
  return groups;
}

export function getLegalMoves(game: Big2Game, player: PlayerIndex): Move[] {
  const hand = game.getHand(player);
  const moves: Move[] = [];

  // First move must contain 3♦ (card 0)
  if (game.isFirstMove()) {
    moves.push(...generateFirstMoves(hand, player));
    return moves;
  }

  const lastMove = game.getLastMove();

  // If there's a last move, can pass
  if (lastMove !== null) {
    moves.push(createMove([], player));
  }

  // If no last move (starting new trick), can play anything
  if (lastMove === null) {
    moves.push(...generateAllMoves(hand, player));
  } else {
    // Must beat last move
    moves.push(...generateBeatingMoves(hand, lastMove.cards, player));
  }

  return moves;
}

function generateFirstMoves(hand: Card[], player: PlayerIndex): Move[] {
  const moves: Move[] = [];

  // Must have 3♦ (card 0)
  if (!hand.includes(0)) {
    return moves;
  }

  // Single 3♦
  moves.push(createMove([0], player));

  // Group cards by rank
  const cardsByRank = groupByRank(hand);

  // Pairs with 3
  const threes = cardsByRank.get(0) || [];
  if (threes.length >= 2) {
    for (const combo of combinations(threes, 2)) {
      moves.push(createMove(combo, player));
    }
  }

  // Triples with 3
  if (threes.length >= 3) {
    for (const combo of combinations(threes, 3)) {
      moves.push(createMove(combo, player));
    }
  }

  // 5-card combos containing 3♦
  if (hand.length >= 5) {
    for (const combo of combinations(hand, 5)) {
      if (combo.includes(0)) {
        const moveType = detectMoveType(combo);
        if (
          moveType === MoveType.STRAIGHT ||
          moveType === MoveType.FLUSH ||
          moveType === MoveType.FULL_HOUSE ||
          moveType === MoveType.QUAD_WITH_KICKER ||
          moveType === MoveType.STRAIGHT_FLUSH
        ) {
          moves.push(createMove(combo, player));
        }
      }
    }
  }

  return moves;
}

function generateAllMoves(hand: Card[], player: PlayerIndex): Move[] {
  const moves: Move[] = [];

  // Singles
  for (const card of hand) {
    moves.push(createMove([card], player));
  }

  // Group cards by rank
  const cardsByRank = groupByRank(hand);

  // Pairs
  for (const cards of cardsByRank.values()) {
    if (cards.length >= 2) {
      for (const combo of combinations(cards, 2)) {
        moves.push(createMove(combo, player));
      }
    }
  }

  // Triples
  for (const cards of cardsByRank.values()) {
    if (cards.length >= 3) {
      for (const combo of combinations(cards, 3)) {
        moves.push(createMove(combo, player));
      }
    }
  }

  // 5-card combinations
  if (hand.length >= 5) {
    for (const combo of combinations(hand, 5)) {
      const moveType = detectMoveType(combo);
      if (
        moveType === MoveType.STRAIGHT ||
        moveType === MoveType.FLUSH ||
        moveType === MoveType.FULL_HOUSE ||
        moveType === MoveType.QUAD_WITH_KICKER ||
        moveType === MoveType.STRAIGHT_FLUSH
      ) {
        moves.push(createMove(combo, player));
      }
    }
  }

  return moves;
}

function generateBeatingMoves(hand: Card[], lastCards: Card[], player: PlayerIndex): Move[] {
  const moves: Move[] = [];
  const lastLen = lastCards.length;

  if (lastLen === 1) {
    // Must play a higher single
    for (const card of hand) {
      if (canBeat([card], lastCards)) {
        moves.push(createMove([card], player));
      }
    }
  } else if (lastLen === 2) {
    // Must play a higher pair
    const cardsByRank = groupByRank(hand);

    for (const cards of cardsByRank.values()) {
      if (cards.length >= 2) {
        for (const combo of combinations(cards, 2)) {
          if (canBeat(combo, lastCards)) {
            moves.push(createMove(combo, player));
          }
        }
      }
    }
  } else if (lastLen === 3) {
    // Must play a higher triple
    const cardsByRank = groupByRank(hand);

    for (const cards of cardsByRank.values()) {
      if (cards.length >= 3) {
        for (const combo of combinations(cards, 3)) {
          if (canBeat(combo, lastCards)) {
            moves.push(createMove(combo, player));
          }
        }
      }
    }
  } else if (lastLen === 5) {
    // Must play a higher 5-card combo
    if (hand.length >= 5) {
      for (const combo of combinations(hand, 5)) {
        if (canBeat(combo, lastCards)) {
          moves.push(createMove(combo, player));
        }
      }
    }
  }

  return moves;
}

// Export combinations for use by greedy bot
export { combinations, groupByRank };
