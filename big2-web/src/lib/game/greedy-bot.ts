/**
 * Enhanced greedy bot agent for Big 2 - TypeScript port of greedy_bot.py
 *
 * Strategy:
 * - Analyze hand composition (pairs, triples, straights, flushes)
 * - Build game state (opponent positions, hand sizes, last move strength)
 * - Score all legal moves based on multiple factors
 * - Intelligent passing based on game state
 */

import { Card, Move, PlayerIndex, MoveType, Rank, HandAnalysis, GameStateInfo } from './types';
import { cardRank, cardSuit } from './constants';
import { Big2Game, isPass } from './game-engine';
import { detectMoveType } from './move-detector';
import { getLegalMoves, combinations } from './move-generator';

// =============================================================================
// Main Entry Point
// =============================================================================

export function selectActionGreedyBot(game: Big2Game, player: PlayerIndex): Move {
  const legalMoves = getLegalMoves(game, player);

  if (legalMoves.length === 0) {
    throw new Error(`No legal moves for player ${player}`);
  }

  // If only one move, must take it
  if (legalMoves.length === 1) {
    return legalMoves[0];
  }

  // Build game state
  const gameState = buildGameState(game, player);

  // Select best move using heuristics
  return selectBestMove(legalMoves, game.getHand(player), gameState);
}

// =============================================================================
// Phase 1: Helper Functions for Enhanced Greedy Bot
// =============================================================================

function findStraights(hand: Card[]): Card[][] {
  const straights: Card[][] = [];

  // Try all 5-card combinations
  for (const combo of combinations(hand, 5)) {
    const ranks = combo.map(c => cardRank(c)).sort((a, b) => a - b);

    // Check if consecutive
    let isStraight = true;
    for (let i = 0; i < 4; i++) {
      if (ranks[i + 1] !== ranks[i] + 1) {
        isStraight = false;
        break;
      }
    }

    if (isStraight) {
      straights.push(combo);
    }
  }

  return straights;
}

function findFlushes(hand: Card[]): Card[][] {
  // Group cards by suit
  const suitCards = new Map<number, Card[]>();
  for (const card of hand) {
    const suit = cardSuit(card);
    if (!suitCards.has(suit)) {
      suitCards.set(suit, []);
    }
    suitCards.get(suit)!.push(card);
  }

  const flushes: Card[][] = [];

  // Find 5-card flushes in each suit
  for (const cards of suitCards.values()) {
    if (cards.length >= 5) {
      for (const combo of combinations(cards, 5)) {
        flushes.push(combo);
      }
    }
  }

  return flushes;
}

function analyzeHandComposition(hand: Card[]): HandAnalysis {
  const ranks = hand.map(c => cardRank(c));
  const rankCounts = new Map<number, number>();

  for (const rank of ranks) {
    rankCounts.set(rank, (rankCounts.get(rank) || 0) + 1);
  }

  return {
    pairs: [...rankCounts.entries()].filter(([_, c]) => c === 2).map(([r]) => r) as Rank[],
    triples: [...rankCounts.entries()].filter(([_, c]) => c === 3).map(([r]) => r) as Rank[],
    quads: [...rankCounts.entries()].filter(([_, c]) => c === 4).map(([r]) => r) as Rank[],
    straights: findStraights(hand),
    flushes: findFlushes(hand),
    highCards: hand.filter(c => cardRank(c) >= 10), // K=10, A=11, 2=12
    singles: [...rankCounts.entries()].filter(([_, c]) => c === 1).map(([r]) => r) as Rank[],
  };
}

function buildGameState(game: Big2Game, player: PlayerIndex): GameStateInfo {
  const opponentSizes: number[] = [];
  for (let i = 0; i < 4; i++) {
    if (i !== player) {
      opponentSizes.push(game.getHandSize(i as PlayerIndex));
    }
  }

  return {
    handSize: game.getHandSize(player),
    minOpponentCards: Math.min(...opponentSizes),
    maxOpponentCards: Math.max(...opponentSizes),
    passesSinceLast: game.getPassesSinceLastMove(),
    isFirstMove: game.isFirstMove(),
    lastMove: game.getLastMove(),
    position: player,
    turnsRemaining: opponentSizes.reduce((a, b) => a + b, 0),
  };
}

function evaluateLastMoveStrength(gameState: GameStateInfo): number {
  const lastMove = gameState.lastMove;

  if (!lastMove || isPass(lastMove)) {
    return 0;
  }

  // Get move type
  const moveType = detectMoveType(lastMove.cards);

  // Base strength by move type
  const typeStrength: Record<MoveType, number> = {
    [MoveType.SINGLE]: 0.2,
    [MoveType.PAIR]: 0.3,
    [MoveType.TRIPLE]: 0.4,
    [MoveType.STRAIGHT]: 0.5,
    [MoveType.FLUSH]: 0.6,
    [MoveType.FULL_HOUSE]: 0.7,
    [MoveType.QUAD_WITH_KICKER]: 0.8,
    [MoveType.STRAIGHT_FLUSH]: 0.9,
    [MoveType.PASS]: 0,
    [MoveType.INVALID]: 0,
  };

  const baseStrength = typeStrength[moveType] || 0;

  // Adjust based on card ranks
  const ranks = lastMove.cards.map(c => cardRank(c));
  const avgRank = ranks.reduce((a, b) => a + b, 0) / ranks.length;
  const rankStrength = avgRank / 12.0; // Normalize to 0-1

  // Combine (70% type, 30% rank)
  return 0.7 * baseStrength + 0.3 * rankStrength;
}

function breaksCombination(move: Move, handAnalysis: HandAnalysis): boolean {
  if (isPass(move)) {
    return false;
  }

  const moveCards = new Set(move.cards);
  const moveType = detectMoveType(move.cards);
  const moveRanks = move.cards.map(c => cardRank(c));

  // Playing single from a pair
  if (moveType === MoveType.SINGLE) {
    const singleRank = moveRanks[0];

    if (handAnalysis.pairs.includes(singleRank as Rank)) {
      return true;
    }

    // Check if breaking a triple
    if (handAnalysis.triples.includes(singleRank as Rank)) {
      return true;
    }

    // Check if breaking a straight
    for (const straight of handAnalysis.straights) {
      if (straight.some(c => moveCards.has(c))) {
        return true;
      }
    }

    // Check if breaking a flush
    for (const flush of handAnalysis.flushes) {
      if (flush.some(c => moveCards.has(c))) {
        return true;
      }
    }
  }

  // Playing pair from triple
  if (moveType === MoveType.PAIR) {
    const pairRank = moveRanks[0];
    if (handAnalysis.triples.includes(pairRank as Rank)) {
      return true;
    }
  }

  return false;
}

// =============================================================================
// Phase 2: Scoring System
// =============================================================================

function hasHighCard(move: Move): boolean {
  if (isPass(move)) {
    return false;
  }

  return move.cards.some(c => cardRank(c) >= 10);
}

function maxRankInMove(move: Move): number {
  if (isPass(move)) {
    return -1;
  }

  return Math.max(...move.cards.map(c => cardRank(c)));
}

function wouldWasteCombo(legalMoves: Move[], handAnalysis: HandAnalysis): boolean {
  const nonPass = legalMoves.filter(m => !isPass(m));

  if (nonPass.length === 0) {
    return false;
  }

  // Check if any non-pass move would break a combo
  for (const move of nonPass) {
    if (breaksCombination(move, handAnalysis)) {
      return true;
    }
  }

  return false;
}

function scorePass(gameState: GameStateInfo): number {
  let score = 0;

  // Pass gets base score
  score += 20.0;

  // Bonus if last move was strong
  const lastStrength = evaluateLastMoveStrength(gameState);
  score += lastStrength * 30.0;

  // Bonus if we have many cards (conservative play)
  if (gameState.handSize > 8) {
    score += 20.0;
  }

  return score;
}

function scoreMove(move: Move, handAnalysis: HandAnalysis, gameState: GameStateInfo): number {
  if (isPass(move)) {
    return scorePass(gameState);
  }

  let score = 0;
  const moveType = detectMoveType(move.cards);

  // Factor 1: Prefer playing low cards early/mid game (dominant factor)
  const avgRank = move.cards.reduce((a, c) => a + cardRank(c), 0) / move.cards.length;
  score += (12 - avgRank) * 15; // Increased to make this the primary driver

  // Factor 2: Preserve valuable combinations (reduced significantly)
  if (breaksCombination(move, handAnalysis)) {
    score -= 15; // Reduced from 35 - less combo preservation
  }

  // Factor 3: Prefer multi-card plays to reduce hand size faster
  score += move.cards.length * 3; // Reduced from 4

  // Factor 4: End-game urgency (few cards left)
  if (gameState.handSize <= 3) {
    score += 50; // Reduced from 80
  }

  // Factor 5: Avoid wasting power cards on weak plays
  const lastStrength = evaluateLastMoveStrength(gameState);
  if (hasHighCard(move) && lastStrength < 0.5) {
    score -= 10; // Reduced from 20
  }

  // Factor 6: Bonus for playing strong combos (straights, flushes, etc.)
  if (
    moveType === MoveType.STRAIGHT ||
    moveType === MoveType.FLUSH ||
    moveType === MoveType.FULL_HOUSE ||
    moveType === MoveType.QUAD_WITH_KICKER ||
    moveType === MoveType.STRAIGHT_FLUSH
  ) {
    score += 20; // Reduced from 30
  }

  // Factor 7: When opponent is close to winning, play aggressively
  if (gameState.minOpponentCards <= 2) {
    score += 25; // Reduced from 40
  }

  return score;
}

// =============================================================================
// Phase 3: Decision Logic
// =============================================================================

function shouldPass(legalMoves: Move[], gameState: GameStateInfo, handAnalysis: HandAnalysis): boolean {
  // Never pass if about to win (<=2 cards)
  if (gameState.handSize <= 2) {
    return false;
  }

  // Never pass if we can win this turn
  const nonPass = legalMoves.filter(m => !isPass(m));
  if (nonPass.some(m => m.cards.length === gameState.handSize)) {
    return false;
  }

  // Pass if opponent is close to winning and we can't block effectively
  if (gameState.minOpponentCards <= 2) {
    if (nonPass.length > 0) {
      const bestPlay = nonPass.reduce((best, m) =>
        maxRankInMove(m) > maxRankInMove(best) ? m : best
      , nonPass[0]);
      if (maxRankInMove(bestPlay) < 9) {
        return true;
      }
    }
  }

  // Pass if we'd waste a valuable combo on a strong play
  const lastStrength = evaluateLastMoveStrength(gameState);
  if (lastStrength > 0.75) {
    if (wouldWasteCombo(legalMoves, handAnalysis)) {
      return true;
    }
  }

  // Pass conservatively in early game (>9 cards instead of >8)
  if (gameState.handSize > 9) {
    if (lastStrength > 0.55) {
      return true;
    }
  }

  return false;
}

function selectBestMove(legalMoves: Move[], hand: Card[], gameState: GameStateInfo): Move {
  // Analyze hand
  const handAnalysis = analyzeHandComposition(hand);

  // Check if we should pass
  const passMove = legalMoves.find(m => isPass(m));
  if (passMove && shouldPass(legalMoves, gameState, handAnalysis)) {
    return passMove;
  }

  // Score all non-pass moves
  const nonPassMoves = legalMoves.filter(m => !isPass(m));
  if (nonPassMoves.length === 0) {
    return passMove!;
  }

  // Score each move
  const moveScores: [Move, number][] = nonPassMoves.map(m => [m, scoreMove(m, handAnalysis, gameState)]);

  // Select highest scoring move
  moveScores.sort((a, b) => b[1] - a[1]);
  return moveScores[0][0];
}

// Export for use in web app
export { analyzeHandComposition, buildGameState };
