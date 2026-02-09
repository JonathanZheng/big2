'use client';

import { cn } from '@/lib/utils';
import { Card, MoveType } from '@/lib/game/types';
import { RANK_NAMES, SUIT_NAMES, cardRank, cardSuit } from '@/lib/game/constants';
import { getMoveTypeName } from '@/lib/game/move-detector';

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

interface SuggestionDisplayProps {
  suggestion: MoveSuggestion | null;
  isLoading?: boolean;
  error?: string | null;
}

export function SuggestionDisplay({
  suggestion,
  isLoading = false,
  error = null,
}: SuggestionDisplayProps) {
  const getSuitColor = (suit: number) => {
    return suit === 0 || suit === 2 ? 'text-red-600' : 'text-gray-900';
  };

  const renderCards = (cards: Card[], size: 'lg' | 'sm' = 'lg') => {
    if (cards.length === 0) {
      return (
        <span
          className={cn(
            'italic text-gray-500',
            size === 'lg' ? 'text-xl' : 'text-sm'
          )}
        >
          Pass
        </span>
      );
    }

    return (
      <div className={cn('flex gap-1', size === 'lg' ? 'gap-2' : 'gap-1')}>
        {[...cards]
          .sort((a, b) => a - b)
          .map((card) => (
            <div
              key={card}
              className={cn(
                'bg-white rounded shadow font-bold flex flex-col items-center justify-center',
                size === 'lg' ? 'w-12 h-16 text-lg' : 'w-8 h-10 text-xs',
                getSuitColor(cardSuit(card))
              )}
            >
              <span>{RANK_NAMES[cardRank(card)]}</span>
              <span className={size === 'lg' ? 'text-xl' : 'text-sm'}>
                {SUIT_NAMES[cardSuit(card)]}
              </span>
            </div>
          ))}
      </div>
    );
  };

  if (isLoading) {
    return (
      <div className="bg-white/10 rounded-lg p-6">
        <div className="flex items-center justify-center gap-3 text-white">
          <div className="animate-spin w-6 h-6 border-2 border-white/30 border-t-white rounded-full" />
          <span>Analyzing position...</span>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-500/20 border border-red-500/30 rounded-lg p-6">
        <h3 className="text-red-300 font-medium mb-2">Error</h3>
        <p className="text-red-200 text-sm">{error}</p>
      </div>
    );
  }

  if (!suggestion) {
    return (
      <div className="bg-white/5 rounded-lg p-6 text-center">
        <p className="text-white/50">
          Select your 13-card hand and click "Analyze" to get AI suggestions
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Main suggestion */}
      <div className="bg-gradient-to-br from-green-600/20 to-green-700/20 border border-green-500/30 rounded-lg p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-green-300 font-medium text-lg">Recommended Move</h3>
          <span className="text-green-400 text-sm bg-green-500/20 px-2 py-1 rounded">
            {getMoveTypeName(suggestion.moveType)}
          </span>
        </div>

        <div className="flex justify-center mb-4">{renderCards(suggestion.cards, 'lg')}</div>

        <div className="bg-black/20 rounded-lg p-4 mt-4">
          <h4 className="text-white/70 text-sm mb-2">Why this move?</h4>
          <p className="text-white text-sm leading-relaxed">{suggestion.reasoning}</p>
        </div>
      </div>

      {/* Alternative moves */}
      {suggestion.alternatives.length > 0 && (
        <div className="bg-white/5 rounded-lg p-4">
          <h3 className="text-white/70 font-medium mb-3">Alternative Moves</h3>
          <div className="space-y-3">
            {suggestion.alternatives.slice(0, 5).map((alt, index) => (
              <div
                key={index}
                className="flex items-center justify-between p-3 bg-white/5 rounded-lg"
              >
                <div className="flex items-center gap-3">
                  <span className="text-white/40 text-sm w-6">#{index + 2}</span>
                  {renderCards(alt.cards, 'sm')}
                </div>
                <span className="text-white/50 text-xs">
                  {getMoveTypeName(alt.moveType)}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
