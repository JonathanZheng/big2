/**
 * Constants for Big 2 card game
 */

export const RANK_NAMES = ['3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A', '2'] as const;
export const SUIT_NAMES = ['♦', '♣', '♥', '♠'] as const;
export const SUIT_SYMBOLS = ['d', 'c', 'h', 's'] as const;
export const SUIT_FULL_NAMES = ['Diamonds', 'Clubs', 'Hearts', 'Spades'] as const;

// Card utilities
export function cardRank(card: number): number {
  return Math.floor(card / 4);
}

export function cardSuit(card: number): number {
  return card % 4;
}

export function cardToStr(card: number): string {
  return `${RANK_NAMES[cardRank(card)]}${SUIT_NAMES[cardSuit(card)]}`;
}

export function cardsToStr(cards: number[]): string {
  const sorted = [...cards].sort((a, b) => a - b);
  return '[' + sorted.map(c => cardToStr(c)).join(', ') + ']';
}

// Create a card from rank and suit
export function makeCard(rank: number, suit: number): number {
  return rank * 4 + suit;
}

// Get all 52 cards
export function getAllCards(): number[] {
  return Array.from({ length: 52 }, (_, i) => i);
}

// Parse card string like "3d" or "As" to card number
export function parseCard(str: string): number | null {
  const rankStr = str.slice(0, -1).toUpperCase();
  const suitStr = str.slice(-1).toLowerCase();

  const rankIndex = RANK_NAMES.findIndex(r => r === rankStr);
  const suitIndex = SUIT_SYMBOLS.findIndex(s => s === suitStr);

  if (rankIndex === -1 || suitIndex === -1) {
    return null;
  }

  return makeCard(rankIndex, suitIndex);
}
