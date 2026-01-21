'use client';

import { cn } from '@/lib/utils';
import { Card } from './Card';
import { Move, detectMoveType, getMoveTypeName } from '@/lib/game';

interface PlayAreaProps {
  lastMove: Move | null;
  message?: string;
}

export function PlayArea({ lastMove, message }: PlayAreaProps) {
  const moveType = lastMove ? detectMoveType(lastMove.cards) : null;
  const moveTypeName = moveType !== null ? getMoveTypeName(moveType) : null;

  return (
    <div className="flex flex-col items-center justify-center gap-4 p-8 min-h-[200px]">
      {lastMove && lastMove.cards.length > 0 ? (
        <>
          <div className="flex gap-1">
            {lastMove.cards.sort((a, b) => a - b).map((card) => (
              <Card key={card} card={card} size="md" disabled />
            ))}
          </div>
          <div className="text-sm text-gray-500">
            Player {lastMove.player} played {moveTypeName}
          </div>
        </>
      ) : (
        <div className="text-gray-400 text-lg">
          {message || 'Play Area'}
        </div>
      )}
    </div>
  );
}

interface SelectedCardsPreviewProps {
  cards: number[];
}

export function SelectedCardsPreview({ cards }: SelectedCardsPreviewProps) {
  if (cards.length === 0) {
    return (
      <div className="text-gray-400 text-sm h-10 flex items-center">
        Select cards to play
      </div>
    );
  }

  const sortedCards = [...cards].sort((a, b) => a - b);
  const moveType = detectMoveType(sortedCards);
  const moveTypeName = getMoveTypeName(moveType);

  return (
    <div className="flex flex-col items-center gap-2">
      <div className="flex gap-1">
        {sortedCards.map((card) => (
          <Card key={card} card={card} size="sm" disabled />
        ))}
      </div>
      <div className={cn(
        'text-sm font-medium',
        moveTypeName === 'Invalid' ? 'text-red-500' : 'text-green-600'
      )}>
        {moveTypeName}
      </div>
    </div>
  );
}
