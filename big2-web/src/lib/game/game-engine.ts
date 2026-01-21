/**
 * Big 2 game engine - TypeScript port of game.py
 *
 * Rules:
 * - 52 cards, 4 players, 13 cards each
 * - First move must contain 3♦
 * - Move types: Single, Pair, Triple, Straight(5), Flush(5), Full House, Quad+Kicker, Straight Flush
 * - Pass allowed after any non-pass move
 * - Game ends when any player runs out of cards
 */

import { Card, Move, PlayerIndex, GameState, StepResult } from './types';
import { cardRank, cardsToStr } from './constants';

export function createMove(cards: Card[], player: PlayerIndex): Move {
  return { cards, player };
}

export function isPass(move: Move): boolean {
  return move.cards.length === 0;
}

export function moveToString(move: Move): string {
  if (isPass(move)) {
    return `Move(PASS, player=${move.player})`;
  }
  return `Move(${cardsToStr(move.cards)}, player=${move.player})`;
}

// Fisher-Yates shuffle
function shuffle<T>(array: T[], seed?: number): T[] {
  const result = [...array];

  // Simple seeded random number generator
  let random: () => number;
  if (seed !== undefined) {
    let s = seed;
    random = () => {
      s = (s * 1103515245 + 12345) & 0x7fffffff;
      return s / 0x7fffffff;
    };
  } else {
    random = Math.random;
  }

  for (let i = result.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [result[i], result[j]] = [result[j], result[i]];
  }
  return result;
}

export class Big2Game {
  private state: GameState;
  private seed?: number;

  constructor(seed?: number) {
    this.seed = seed;
    this.state = this.createInitialState();
  }

  private createInitialState(): GameState {
    // Create and shuffle deck
    const deck = shuffle(Array.from({ length: 52 }, (_, i) => i), this.seed);

    // Deal cards to players
    const hands: Card[][] = [
      deck.slice(0, 13).sort((a, b) => a - b),
      deck.slice(13, 26).sort((a, b) => a - b),
      deck.slice(26, 39).sort((a, b) => a - b),
      deck.slice(39, 52).sort((a, b) => a - b),
    ];

    // Find who has 3♦ (card 0)
    let currentPlayer: PlayerIndex = 0;
    for (let i = 0; i < 4; i++) {
      if (hands[i].includes(0)) {
        currentPlayer = i as PlayerIndex;
        break;
      }
    }

    return {
      hands,
      currentPlayer,
      firstMove: true,
      lastMove: null,
      passesSinceLastMove: 0,
      done: false,
      winner: null,
      moveHistory: [],
    };
  }

  reset(): void {
    this.state = this.createInitialState();
  }

  getState(): GameState {
    return { ...this.state };
  }

  getHand(player: PlayerIndex): Card[] {
    return [...this.state.hands[player]];
  }

  getHandSize(player: PlayerIndex): number {
    return this.state.hands[player].length;
  }

  getCurrentPlayer(): PlayerIndex {
    return this.state.currentPlayer;
  }

  getLastMove(): Move | null {
    return this.state.lastMove;
  }

  isFirstMove(): boolean {
    return this.state.firstMove;
  }

  isDone(): boolean {
    return this.state.done;
  }

  getWinner(): PlayerIndex | null {
    return this.state.winner;
  }

  getPassesSinceLastMove(): number {
    return this.state.passesSinceLastMove;
  }

  getMoveHistory(): Move[] {
    return [...this.state.moveHistory];
  }

  step(move: Move): StepResult {
    if (this.state.done) {
      throw new Error('Game is already over');
    }

    if (move.player !== this.state.currentPlayer) {
      throw new Error(`Wrong player: expected ${this.state.currentPlayer}, got ${move.player}`);
    }

    if (!isPass(move)) {
      // Remove cards from hand
      for (const card of move.cards) {
        const idx = this.state.hands[this.state.currentPlayer].indexOf(card);
        if (idx === -1) {
          throw new Error(`Card ${card} not in player's hand`);
        }
        this.state.hands[this.state.currentPlayer].splice(idx, 1);
      }

      // Update last move
      this.state.lastMove = move;
      this.state.passesSinceLastMove = 0;
      this.state.firstMove = false;

      // Check if player won
      if (this.state.hands[this.state.currentPlayer].length === 0) {
        this.state.done = true;
        this.state.winner = this.state.currentPlayer;
      }
    } else {
      // Pass
      this.state.passesSinceLastMove++;

      // If 3 consecutive passes, start new trick
      if (this.state.passesSinceLastMove >= 3) {
        this.state.lastMove = null;
        this.state.passesSinceLastMove = 0;
      }
    }

    // Record move
    this.state.moveHistory.push(move);

    // Next player
    this.state.currentPlayer = ((this.state.currentPlayer + 1) % 4) as PlayerIndex;

    // Compute rewards if game is done
    if (this.state.done) {
      const rewards = this.computeRewards();
      return {
        reward: rewards[move.player],
        done: true,
        info: {
          allRewards: rewards,
          winner: this.state.winner!,
        },
      };
    }

    return {
      reward: 0,
      done: false,
      info: {},
    };
  }

  private computeRewards(): number[] {
    if (!this.state.done || this.state.winner === null) {
      throw new Error('Game must be over to compute rewards');
    }

    const rewards = [-1, -1, -1, -1];
    rewards[this.state.winner] = 1;
    return rewards;
  }

  toString(): string {
    const lines = [`Big2Game(currentPlayer=${this.state.currentPlayer}, done=${this.state.done})`];

    for (let i = 0; i < 4; i++) {
      const handStr = cardsToStr(this.state.hands[i]);
      const marker = i === this.state.currentPlayer ? ' ←' : '';
      lines.push(`  Player ${i}: ${handStr}${marker}`);
    }

    if (this.state.lastMove) {
      lines.push(`  Last move: ${moveToString(this.state.lastMove)}`);
    }

    return lines.join('\n');
  }
}

// Export a factory function for creating games
export function createGame(seed?: number): Big2Game {
  return new Big2Game(seed);
}
