/**
 * Type definitions for Big 2 card game
 */

// Card representation:
// Card = rank * 4 + suit
// Rank: 0=3, 1=4, 2=5, ..., 10=K, 11=A, 12=2
// Suit: 0=Diamonds, 1=Clubs, 2=Hearts, 3=Spades
// So 3♦ = 0*4 + 0 = 0

export type Card = number; // 0-51

export type Suit = 0 | 1 | 2 | 3; // Diamonds, Clubs, Hearts, Spades
export type Rank = 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12;

export type PlayerIndex = 0 | 1 | 2 | 3;

export enum MoveType {
  PASS = 0,
  SINGLE = 1,
  PAIR = 2,
  TRIPLE = 3,
  STRAIGHT = 4,
  FLUSH = 5,
  FULL_HOUSE = 6,
  QUAD_WITH_KICKER = 7,
  STRAIGHT_FLUSH = 8,
  INVALID = 9,
}

export interface Move {
  cards: Card[];
  player: PlayerIndex;
}

export interface GameState {
  hands: Card[][];
  currentPlayer: PlayerIndex;
  firstMove: boolean;
  lastMove: Move | null;
  passesSinceLastMove: number;
  done: boolean;
  winner: PlayerIndex | null;
  moveHistory: Move[];
}

export interface StepResult {
  reward: number;
  done: boolean;
  info: {
    allRewards?: number[];
    winner?: PlayerIndex;
  };
}

export interface HandAnalysis {
  pairs: Rank[];
  triples: Rank[];
  quads: Rank[];
  straights: Card[][];
  flushes: Card[][];
  highCards: Card[];
  singles: Rank[];
}

export interface GameStateInfo {
  handSize: number;
  minOpponentCards: number;
  maxOpponentCards: number;
  passesSinceLast: number;
  isFirstMove: boolean;
  lastMove: Move | null;
  position: PlayerIndex;
  turnsRemaining: number;
}
