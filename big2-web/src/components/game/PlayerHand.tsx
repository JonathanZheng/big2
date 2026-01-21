'use client';

import { cn } from '@/lib/utils';
import { Card, CardBack } from './Card';

interface PlayerHandProps {
  cards: number[];
  selectedCards: number[];
  onCardClick?: (card: number) => void;
  disabled?: boolean;
  isCurrentPlayer?: boolean;
  position: 'bottom' | 'left' | 'top' | 'right';
  playerName?: string;
  showCards?: boolean;
}

export function PlayerHand({
  cards,
  selectedCards,
  onCardClick,
  disabled = false,
  isCurrentPlayer = false,
  position,
  playerName,
  showCards = true,
}: PlayerHandProps) {
  const sortedCards = [...cards].sort((a, b) => a - b);

  // Position-specific layouts
  const containerClasses = {
    bottom: 'flex-row',
    top: 'flex-row',
    left: 'flex-col',
    right: 'flex-col',
  };

  const cardOverlap = {
    bottom: '-ml-8 first:ml-0',
    top: '-ml-8 first:ml-0',
    left: '-mt-12 first:mt-0',
    right: '-mt-12 first:mt-0',
  };

  return (
    <div className={cn('flex flex-col items-center gap-2', position === 'left' || position === 'right' ? 'flex-row' : '')}>
      {playerName && (
        <div
          className={cn(
            'px-3 py-1 rounded-full text-sm font-medium',
            isCurrentPlayer ? 'bg-blue-500 text-white' : 'bg-gray-200 text-gray-700'
          )}
        >
          {playerName}
          {isCurrentPlayer && <span className="ml-1 animate-pulse">•</span>}
        </div>
      )}

      <div className={cn('flex', containerClasses[position])}>
        {sortedCards.map((card) =>
          showCards ? (
            <div key={card} className={cn(cardOverlap[position])}>
              <Card
                card={card}
                selected={selectedCards.includes(card)}
                onClick={() => onCardClick?.(card)}
                disabled={disabled}
                size={position === 'bottom' ? 'lg' : 'sm'}
              />
            </div>
          ) : (
            <div key={card} className={cn(cardOverlap[position])}>
              <CardBack size={position === 'bottom' ? 'lg' : 'sm'} />
            </div>
          )
        )}
      </div>

      {!showCards && cards.length > 0 && (
        <div className="text-sm text-gray-500">{cards.length} cards</div>
      )}
    </div>
  );
}
