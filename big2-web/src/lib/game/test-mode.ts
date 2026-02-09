/**
 * Test Mode utilities for Big 2 AI analysis
 *
 * Allows users to input their hand and game history to get AI suggestions
 */

import { Card, Move, PlayerIndex, MoveType, GameState } from './types';
import { cardRank, cardSuit, cardsToStr, RANK_NAMES, SUIT_NAMES } from './constants';
import { detectMoveType, getMoveTypeName, canBeat } from './move-detector';
import { getLegalMoves } from './move-generator';
import { isPass } from './game-engine';

// Types for test mode
export interface TestModeMove {
  player: PlayerIndex;
  cards: Card[];
  isPass: boolean;
}

export interface TestModeState {
  myHand: Card[];
  myPosition: PlayerIndex;
  history: TestModeMove[];
}

export interface MoveSuggestion {
  cards: Card[];
  moveType: MoveType;
  reasoning: string;
  score: number;
  alternatives: {
    cards: Card[];
    moveType: MoveType;
    score: number;
  }[];
}

// Mock game class for test mode that allows custom state
class TestModeGame {
  private hands: Card[][];
  private currentPlayer: PlayerIndex;
  private firstMove: boolean;
  private lastMove: Move | null;
  private passesSinceLastMove: number;
  private graveyard: Set<Card>;

  constructor(
    myHand: Card[],
    myPosition: PlayerIndex,
    history: TestModeMove[]
  ) {
    // Initialize with 13 cards per player (unknown cards for opponents)
    // Cards in myHand and graveyard are accounted for
    this.graveyard = new Set<Card>();
    this.hands = [[], [], [], []];
    this.hands[myPosition] = [...myHand].sort((a, b) => a - b);

    // Process history to build graveyard and determine current state
    this.firstMove = history.length === 0;
    this.lastMove = null;
    this.passesSinceLastMove = 0;
    this.currentPlayer = 0; // Will be updated based on history

    // Find who has 3♦ (card 0) - determines starting player
    if (myHand.includes(0)) {
      this.currentPlayer = myPosition;
    } else {
      // Check history to see who played 3♦
      for (const move of history) {
        if (move.cards.includes(0)) {
          // This player started
          break;
        }
      }
    }

    // Process history
    let consecutivePasses = 0;
    for (const move of history) {
      if (move.isPass) {
        consecutivePasses++;
        if (consecutivePasses >= 3) {
          this.lastMove = null;
          consecutivePasses = 0;
        }
      } else {
        consecutivePasses = 0;
        this.lastMove = { cards: move.cards, player: move.player };
        this.firstMove = false;

        // Add to graveyard
        for (const card of move.cards) {
          this.graveyard.add(card);
        }
      }
    }

    this.passesSinceLastMove = consecutivePasses;

    // Determine current player based on history
    if (history.length > 0) {
      const lastMovePlayer = history[history.length - 1].player;
      this.currentPlayer = ((lastMovePlayer + 1) % 4) as PlayerIndex;
    } else {
      // First move - find who has 3♦
      this.currentPlayer = myHand.includes(0) ? myPosition : 0;
    }

    // Fill opponent hands with remaining cards (for hand size estimation)
    const usedCards = new Set([...myHand, ...this.graveyard]);
    const remainingCards = Array.from({ length: 52 }, (_, i) => i).filter(
      (c) => !usedCards.has(c)
    );

    // Distribute remaining cards evenly to opponents
    let cardIndex = 0;
    for (let p = 0; p < 4; p++) {
      if (p !== myPosition) {
        // Estimate cards based on history - each non-pass move removes cards
        const opponentMoves = history.filter((m) => m.player === p && !m.isPass);
        const cardsPlayed = opponentMoves.reduce((sum, m) => sum + m.cards.length, 0);
        const estimatedCards = 13 - cardsPlayed;

        // Assign placeholder cards
        for (let i = 0; i < estimatedCards && cardIndex < remainingCards.length; i++) {
          this.hands[p].push(remainingCards[cardIndex++]);
        }
      }
    }
  }

  getHand(player: PlayerIndex): Card[] {
    return [...this.hands[player]];
  }

  getHandSize(player: PlayerIndex): number {
    return this.hands[player].length;
  }

  getCurrentPlayer(): PlayerIndex {
    return this.currentPlayer;
  }

  getLastMove(): Move | null {
    return this.lastMove;
  }

  isFirstMove(): boolean {
    return this.firstMove;
  }

  getPassesSinceLastMove(): number {
    return this.passesSinceLastMove;
  }

  getGraveyard(): Card[] {
    return [...this.graveyard];
  }
}

// Scoring functions (adapted from greedy-bot)
function scoreMove(
  move: Move,
  hand: Card[],
  lastMove: Move | null,
  isFirstMove: boolean,
  handSize: number,
  minOpponentCards: number
): number {
  if (isPass(move)) {
    return 20; // Base pass score
  }

  let score = 0;
  const moveType = detectMoveType(move.cards);

  // Factor 1: Prefer playing low cards
  const avgRank = move.cards.reduce((a, c) => a + cardRank(c), 0) / move.cards.length;
  score += (12 - avgRank) * 15;

  // Factor 2: Multi-card plays
  score += move.cards.length * 3;

  // Factor 3: End-game urgency
  if (handSize <= 3) {
    score += 50;
  }

  // Factor 4: Bonus for strong combos
  if (
    moveType === MoveType.STRAIGHT ||
    moveType === MoveType.FLUSH ||
    moveType === MoveType.FULL_HOUSE ||
    moveType === MoveType.QUAD_WITH_KICKER ||
    moveType === MoveType.STRAIGHT_FLUSH
  ) {
    score += 20;
  }

  // Factor 5: Opponent pressure
  if (minOpponentCards <= 2) {
    score += 25;
  }

  return score;
}

function generateReasoning(
  move: Move,
  hand: Card[],
  lastMove: Move | null,
  isFirstMove: boolean,
  minOpponentCards: number
): string {
  const parts: string[] = [];
  const moveType = detectMoveType(move.cards);

  if (isPass(move)) {
    parts.push('Passing is recommended.');

    if (lastMove) {
      const lastType = detectMoveType(lastMove.cards);
      const lastRanks = lastMove.cards.map((c) => cardRank(c));
      const avgLastRank = lastRanks.reduce((a, b) => a + b, 0) / lastRanks.length;

      if (avgLastRank >= 10) {
        parts.push('The current play is too strong to beat efficiently.');
      }
    }

    if (hand.length > 8) {
      parts.push('With many cards remaining, conserving strong cards is wise.');
    }

    return parts.join(' ');
  }

  // Describe the recommended move
  const cardStr = move.cards
    .sort((a, b) => a - b)
    .map((c) => `${RANK_NAMES[cardRank(c)]}${SUIT_NAMES[cardSuit(c)]}`)
    .join(', ');

  parts.push(`Play ${cardStr} (${getMoveTypeName(moveType)}).`);

  // Add reasoning based on context
  const avgRank = move.cards.reduce((a, c) => a + cardRank(c), 0) / move.cards.length;

  if (isFirstMove && move.cards.includes(0)) {
    parts.push('This includes the required 3\u2666 for the opening move.');
  }

  if (avgRank <= 4) {
    parts.push('Playing low cards early helps preserve stronger cards for later.');
  }

  if (hand.length <= 3) {
    parts.push('With few cards left, playing aggressively to finish is optimal.');
  }

  if (minOpponentCards <= 2) {
    parts.push('An opponent is close to winning - aggressive play is needed.');
  }

  if (
    moveType === MoveType.STRAIGHT ||
    moveType === MoveType.FLUSH ||
    moveType === MoveType.STRAIGHT_FLUSH
  ) {
    parts.push('Strong 5-card combinations are valuable for clearing cards quickly.');
  }

  if (moveType === MoveType.FULL_HOUSE || moveType === MoveType.QUAD_WITH_KICKER) {
    parts.push('This powerful combination is difficult for opponents to beat.');
  }

  return parts.join(' ');
}

// Main analysis function
export function analyzeSituation(state: TestModeState): MoveSuggestion {
  const { myHand, myPosition, history } = state;

  // Validate hand
  if (myHand.length === 0) {
    throw new Error('Please select at least one card in your hand');
  }

  if (myHand.length > 13) {
    throw new Error('Hand cannot have more than 13 cards');
  }

  // Check for duplicate cards
  const cardSet = new Set(myHand);
  if (cardSet.size !== myHand.length) {
    throw new Error('Duplicate cards in hand');
  }

  // Build test game state
  const game = new TestModeGame(myHand, myPosition, history);

  // Check if it's actually the user's turn based on history
  const isMyTurn = game.getCurrentPlayer() === myPosition;

  if (!isMyTurn) {
    throw new Error(
      `Based on the history, it's Player ${game.getCurrentPlayer()}'s turn, not yours (Position ${myPosition}). Add more moves to the history or check your position.`
    );
  }

  // Get legal moves
  const legalMoves = getLegalMovesForTestMode(game, myPosition);

  if (legalMoves.length === 0) {
    throw new Error('No legal moves available - check your input');
  }

  // Calculate opponent stats
  const opponentSizes: number[] = [];
  for (let i = 0; i < 4; i++) {
    if (i !== myPosition) {
      opponentSizes.push(game.getHandSize(i as PlayerIndex));
    }
  }
  const minOpponentCards = Math.min(...opponentSizes);

  // Score all moves
  const moveScores: { move: Move; score: number }[] = legalMoves.map((move) => ({
    move,
    score: scoreMove(
      move,
      myHand,
      game.getLastMove(),
      game.isFirstMove(),
      myHand.length,
      minOpponentCards
    ),
  }));

  // Sort by score descending
  moveScores.sort((a, b) => b.score - a.score);

  const bestMove = moveScores[0].move;
  const bestMoveType = detectMoveType(bestMove.cards);

  // Generate reasoning
  const reasoning = generateReasoning(
    bestMove,
    myHand,
    game.getLastMove(),
    game.isFirstMove(),
    minOpponentCards
  );

  // Get alternatives (skip pass if best is not pass, include pass if best is pass)
  const alternatives = moveScores
    .slice(1, 6)
    .filter((ms) => ms.score > 0)
    .map((ms) => ({
      cards: ms.move.cards,
      moveType: detectMoveType(ms.move.cards),
      score: ms.score,
    }));

  return {
    cards: bestMove.cards,
    moveType: bestMoveType,
    reasoning,
    score: moveScores[0].score,
    alternatives,
  };
}

// Get legal moves for test mode game
function getLegalMovesForTestMode(game: TestModeGame, player: PlayerIndex): Move[] {
  const hand = game.getHand(player);
  const lastMove = game.getLastMove();
  const isFirst = game.isFirstMove();

  const moves: Move[] = [];

  // Pass is always legal unless it's the first move or we control
  if (lastMove !== null && !isFirst) {
    moves.push({ cards: [], player });
  }

  // Generate all possible moves from hand
  const allPossibleMoves = generateAllPossibleMoves(hand, player);

  for (const move of allPossibleMoves) {
    // Validate first move (must include 3♦)
    if (isFirst && !move.cards.includes(0)) {
      continue;
    }

    // Validate move type
    const moveType = detectMoveType(move.cards);
    if (moveType === MoveType.INVALID) {
      continue;
    }

    // Check if can beat last move
    if (lastMove && !canBeat(move.cards, lastMove.cards)) {
      continue;
    }

    moves.push(move);
  }

  // If no non-pass moves and it's not first move, can only pass
  if (moves.length === 0 && !isFirst) {
    moves.push({ cards: [], player });
  }

  return moves;
}

// Generate all possible moves from a hand
function generateAllPossibleMoves(hand: Card[], player: PlayerIndex): Move[] {
  const moves: Move[] = [];

  // Singles
  for (const card of hand) {
    moves.push({ cards: [card], player });
  }

  // Pairs
  const rankGroups = new Map<number, Card[]>();
  for (const card of hand) {
    const rank = cardRank(card);
    if (!rankGroups.has(rank)) {
      rankGroups.set(rank, []);
    }
    rankGroups.get(rank)!.push(card);
  }

  for (const cards of rankGroups.values()) {
    if (cards.length >= 2) {
      for (const combo of combinations(cards, 2)) {
        moves.push({ cards: combo, player });
      }
    }
  }

  // Triples
  for (const cards of rankGroups.values()) {
    if (cards.length >= 3) {
      for (const combo of combinations(cards, 3)) {
        moves.push({ cards: combo, player });
      }
    }
  }

  // 5-card combinations
  if (hand.length >= 5) {
    for (const combo of combinations(hand, 5)) {
      const moveType = detectMoveType(combo);
      if (moveType !== MoveType.INVALID) {
        moves.push({ cards: combo, player });
      }
    }
  }

  return moves;
}

// Combination generator
function combinations<T>(arr: T[], k: number): T[][] {
  if (k > arr.length || k <= 0) {
    return [];
  }

  if (k === arr.length) {
    return [arr.slice()];
  }

  if (k === 1) {
    return arr.map((v) => [v]);
  }

  const result: T[][] = [];

  for (let i = 0; i <= arr.length - k; i++) {
    const head = arr.slice(i, i + 1);
    const tailCombinations = combinations(arr.slice(i + 1), k - 1);
    for (const tail of tailCombinations) {
      result.push(head.concat(tail));
    }
  }

  return result;
}
