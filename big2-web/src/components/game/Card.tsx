'use client';

import { cn } from '@/lib/utils';
import { cardRank, cardSuit, RANK_NAMES, SUIT_NAMES } from '@/lib/game';

interface CardProps {
  card: number;
  selected?: boolean;
  onClick?: () => void;
  disabled?: boolean;
  size?: 'sm' | 'md' | 'lg';
  faceDown?: boolean;
}

export function Card({ card, selected = false, onClick, disabled = false, size = 'md', faceDown = false }: CardProps) {
  const rank = cardRank(card);
  const suit = cardSuit(card);
  const rankName = RANK_NAMES[rank];
  const suitName = SUIT_NAMES[suit];

  // Red suits: Hearts (2), Diamonds (0)
  const isRed = suit === 0 || suit === 2;

  const sizeClasses = {
    sm: 'w-12 h-16 text-sm',
    md: 'w-16 h-22 text-base',
    lg: 'w-20 h-28 text-lg',
  };

  if (faceDown) {
    return (
      <div
        className={cn(
          'rounded-lg border-2 border-gray-300 flex items-center justify-center',
          'bg-gradient-to-br from-blue-600 to-blue-800',
          sizeClasses[size]
        )}
      >
        <div className="text-white/30 text-2xl font-bold">?</div>
      </div>
    );
  }

  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className={cn(
        'rounded-lg border-2 flex flex-col items-center justify-center transition-all',
        'bg-white shadow-md hover:shadow-lg',
        sizeClasses[size],
        selected ? 'border-blue-500 -translate-y-2 ring-2 ring-blue-300' : 'border-gray-300',
        disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer hover:-translate-y-1',
        isRed ? 'text-red-600' : 'text-gray-900'
      )}
    >
      <span className="font-bold leading-none">{rankName}</span>
      <span className="text-xl leading-none">{suitName}</span>
    </button>
  );
}

interface CardBackProps {
  size?: 'sm' | 'md' | 'lg';
}

export function CardBack({ size = 'md' }: CardBackProps) {
  const sizeClasses = {
    sm: 'w-12 h-16',
    md: 'w-16 h-22',
    lg: 'w-20 h-28',
  };

  return (
    <div
      className={cn(
        'rounded-lg border-2 border-gray-400 flex items-center justify-center',
        'bg-gradient-to-br from-blue-600 to-blue-800',
        sizeClasses[size]
      )}
    >
      <div className="w-3/4 h-3/4 rounded border border-white/20 bg-blue-700/50" />
    </div>
  );
}
