'use client';

import { useCallback } from 'react';
import { cn } from '@/lib/utils';
import { RANK_NAMES, SUIT_NAMES, makeCard, cardRank, cardSuit } from '@/lib/game/constants';
import { Card } from '@/lib/game/types';

interface CardSelectorProps {
  selectedCards: Card[];
  onSelectionChange: (cards: Card[]) => void;
  maxCards?: number;
  disabledCards?: Card[];
  title?: string;
}

export function CardSelector({
  selectedCards,
  onSelectionChange,
  maxCards = 13,
  disabledCards = [],
  title = 'Select Your Hand',
}: CardSelectorProps) {
  const handleCardClick = useCallback(
    (card: Card) => {
      if (disabledCards.includes(card)) return;

      if (selectedCards.includes(card)) {
        onSelectionChange(selectedCards.filter((c) => c !== card));
      } else if (selectedCards.length < maxCards) {
        onSelectionChange([...selectedCards, card]);
      }
    },
    [selectedCards, onSelectionChange, maxCards, disabledCards]
  );

  const getSuitColor = (suit: number) => {
    // Diamonds and Hearts are red
    return suit === 0 || suit === 2 ? 'text-red-500' : 'text-gray-900';
  };

  return (
    <div className="bg-white/10 rounded-lg p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-white font-medium">{title}</h3>
        <span
          className={cn(
            'text-sm px-2 py-1 rounded',
            selectedCards.length === maxCards
              ? 'bg-green-500/20 text-green-300'
              : 'bg-white/10 text-white/70'
          )}
        >
          {selectedCards.length} / {maxCards} cards
        </span>
      </div>

      {/* Card grid - 13 ranks x 4 suits */}
      <div className="overflow-x-auto">
        <table className="w-full border-collapse">
          <thead>
            <tr>
              <th className="text-white/50 text-xs p-1"></th>
              {RANK_NAMES.map((rank, i) => (
                <th key={i} className="text-white/70 text-xs p-1 font-medium">
                  {rank}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {[0, 1, 2, 3].map((suit) => (
              <tr key={suit}>
                <td className={cn('text-lg p-1', getSuitColor(suit))}>
                  {SUIT_NAMES[suit]}
                </td>
                {RANK_NAMES.map((_, rankIndex) => {
                  const card = makeCard(rankIndex, suit);
                  const isSelected = selectedCards.includes(card);
                  const isDisabled = disabledCards.includes(card);

                  return (
                    <td key={rankIndex} className="p-0.5">
                      <button
                        onClick={() => handleCardClick(card)}
                        disabled={isDisabled}
                        className={cn(
                          'w-8 h-10 rounded text-xs font-medium transition-all',
                          isDisabled && 'opacity-30 cursor-not-allowed bg-gray-600',
                          !isDisabled && !isSelected && 'bg-white hover:bg-gray-100 hover:scale-105',
                          !isDisabled && isSelected && 'bg-yellow-400 ring-2 ring-yellow-300 scale-105',
                          getSuitColor(suit)
                        )}
                      >
                        {RANK_NAMES[rankIndex]}
                        <br />
                        <span className="text-[10px]">{SUIT_NAMES[suit]}</span>
                      </button>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Selected cards display */}
      {selectedCards.length > 0 && (
        <div className="mt-4 pt-3 border-t border-white/10">
          <div className="text-white/70 text-sm mb-2">Selected:</div>
          <div className="flex flex-wrap gap-1">
            {[...selectedCards]
              .sort((a, b) => a - b)
              .map((card) => (
                <button
                  key={card}
                  onClick={() => handleCardClick(card)}
                  className={cn(
                    'px-2 py-1 rounded text-sm font-medium bg-white hover:bg-red-100 transition-colors',
                    getSuitColor(cardSuit(card))
                  )}
                  title="Click to remove"
                >
                  {RANK_NAMES[cardRank(card)]}
                  {SUIT_NAMES[cardSuit(card)]}
                </button>
              ))}
          </div>
        </div>
      )}

      {selectedCards.length > 0 && (
        <button
          onClick={() => onSelectionChange([])}
          className="mt-3 text-sm text-red-400 hover:text-red-300 underline"
        >
          Clear all
        </button>
      )}
    </div>
  );
}
