'use client';

import { useState, useCallback } from 'react';
import { cn } from '@/lib/utils';
import { Card, PlayerIndex, MoveType } from '@/lib/game/types';
import { RANK_NAMES, SUIT_NAMES, cardRank, cardSuit, makeCard } from '@/lib/game/constants';
import { detectMoveType, getMoveTypeName } from '@/lib/game/move-detector';

export interface HistoryMove {
  player: PlayerIndex;
  cards: Card[];
  isPass: boolean;
}

interface HistoryInputProps {
  history: HistoryMove[];
  onHistoryChange: (history: HistoryMove[]) => void;
  userPosition: PlayerIndex;
  usedCards: Card[]; // Cards already in user's hand - cannot be played by others
}

export function HistoryInput({
  history,
  onHistoryChange,
  userPosition,
  usedCards,
}: HistoryInputProps) {
  const [isAdding, setIsAdding] = useState(false);
  const [currentPlayer, setCurrentPlayer] = useState<PlayerIndex>(0);
  const [selectedCards, setSelectedCards] = useState<Card[]>([]);
  const [isPass, setIsPass] = useState(false);

  // Calculate which cards are already used in history
  const cardsInHistory = history.flatMap((m) => m.cards);
  const unavailableCards = [...usedCards, ...cardsInHistory];

  const handleAddMove = useCallback(() => {
    if (!isAdding) {
      setIsAdding(true);
      return;
    }

    const newMove: HistoryMove = {
      player: currentPlayer,
      cards: isPass ? [] : [...selectedCards].sort((a, b) => a - b),
      isPass,
    };

    onHistoryChange([...history, newMove]);
    setSelectedCards([]);
    setIsPass(false);
    setIsAdding(false);
  }, [isAdding, currentPlayer, selectedCards, isPass, history, onHistoryChange]);

  const handleRemoveMove = useCallback(
    (index: number) => {
      onHistoryChange(history.filter((_, i) => i !== index));
    },
    [history, onHistoryChange]
  );

  const handleCardToggle = useCallback(
    (card: Card) => {
      if (selectedCards.includes(card)) {
        setSelectedCards(selectedCards.filter((c) => c !== card));
      } else {
        setSelectedCards([...selectedCards, card]);
      }
    },
    [selectedCards]
  );

  const getSuitColor = (suit: number) => {
    return suit === 0 || suit === 2 ? 'text-red-500' : 'text-gray-900';
  };

  const getMoveTypeDisplay = (cards: Card[]) => {
    if (cards.length === 0) return 'Pass';
    const type = detectMoveType(cards);
    return getMoveTypeName(type);
  };

  return (
    <div className="bg-white/10 rounded-lg p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-white font-medium">Game History</h3>
        <span className="text-sm text-white/50">{history.length} moves</span>
      </div>

      {/* History list */}
      {history.length > 0 && (
        <div className="space-y-2 mb-4 max-h-48 overflow-y-auto">
          {history.map((move, index) => (
            <div
              key={index}
              className={cn(
                'flex items-center justify-between p-2 rounded text-sm',
                move.player === userPosition
                  ? 'bg-blue-500/20 border border-blue-500/30'
                  : 'bg-white/5'
              )}
            >
              <div className="flex items-center gap-2">
                <span className="text-white/70 w-16">
                  {move.player === userPosition ? 'You' : `Player ${move.player}`}:
                </span>
                {move.isPass ? (
                  <span className="text-gray-400 italic">Pass</span>
                ) : (
                  <div className="flex gap-1">
                    {move.cards.map((card) => (
                      <span
                        key={card}
                        className={cn(
                          'px-1.5 py-0.5 bg-white rounded text-xs font-medium',
                          getSuitColor(cardSuit(card))
                        )}
                      >
                        {RANK_NAMES[cardRank(card)]}
                        {SUIT_NAMES[cardSuit(card)]}
                      </span>
                    ))}
                    <span className="text-white/50 text-xs ml-1">
                      ({getMoveTypeDisplay(move.cards)})
                    </span>
                  </div>
                )}
              </div>
              <button
                onClick={() => handleRemoveMove(index)}
                className="text-red-400 hover:text-red-300 text-xs"
              >
                Remove
              </button>
            </div>
          ))}
        </div>
      )}

      {/* Add move form */}
      {isAdding ? (
        <div className="space-y-3 p-3 bg-white/5 rounded-lg">
          <div className="flex items-center gap-4">
            <label className="text-white/70 text-sm">Player:</label>
            <select
              value={currentPlayer}
              onChange={(e) => setCurrentPlayer(Number(e.target.value) as PlayerIndex)}
              className="bg-white/10 text-white border border-white/20 rounded px-2 py-1 text-sm"
            >
              <option value={0} className="bg-gray-800">
                {userPosition === 0 ? 'You (0)' : 'Player 0'}
              </option>
              <option value={1} className="bg-gray-800">
                {userPosition === 1 ? 'You (1)' : 'Player 1'}
              </option>
              <option value={2} className="bg-gray-800">
                {userPosition === 2 ? 'You (2)' : 'Player 2'}
              </option>
              <option value={3} className="bg-gray-800">
                {userPosition === 3 ? 'You (3)' : 'Player 3'}
              </option>
            </select>

            <label className="flex items-center gap-2 text-white/70 text-sm">
              <input
                type="checkbox"
                checked={isPass}
                onChange={(e) => {
                  setIsPass(e.target.checked);
                  if (e.target.checked) setSelectedCards([]);
                }}
                className="rounded"
              />
              Pass
            </label>
          </div>

          {!isPass && (
            <div>
              <div className="text-white/70 text-sm mb-2">Select cards played:</div>
              <div className="flex flex-wrap gap-1 max-h-32 overflow-y-auto">
                {Array.from({ length: 52 }, (_, i) => i).map((card) => {
                  const isUnavailable = unavailableCards.includes(card);
                  const isSelected = selectedCards.includes(card);

                  return (
                    <button
                      key={card}
                      onClick={() => !isUnavailable && handleCardToggle(card)}
                      disabled={isUnavailable}
                      className={cn(
                        'px-1.5 py-0.5 rounded text-xs font-medium transition-all',
                        isUnavailable && 'opacity-20 cursor-not-allowed bg-gray-500',
                        !isUnavailable && !isSelected && 'bg-white/80 hover:bg-white',
                        !isUnavailable && isSelected && 'bg-yellow-400 ring-1 ring-yellow-300',
                        getSuitColor(cardSuit(card))
                      )}
                    >
                      {RANK_NAMES[cardRank(card)]}
                      {SUIT_NAMES[cardSuit(card)]}
                    </button>
                  );
                })}
              </div>
              {selectedCards.length > 0 && (
                <div className="mt-2 text-white/50 text-xs">
                  {getMoveTypeDisplay(selectedCards)}
                </div>
              )}
            </div>
          )}

          <div className="flex gap-2">
            <button
              onClick={handleAddMove}
              disabled={!isPass && selectedCards.length === 0}
              className="px-3 py-1.5 bg-green-600 hover:bg-green-500 disabled:bg-gray-600 disabled:cursor-not-allowed text-white text-sm rounded transition-colors"
            >
              Add Move
            </button>
            <button
              onClick={() => {
                setIsAdding(false);
                setSelectedCards([]);
                setIsPass(false);
              }}
              className="px-3 py-1.5 bg-white/10 hover:bg-white/20 text-white text-sm rounded transition-colors"
            >
              Cancel
            </button>
          </div>
        </div>
      ) : (
        <button
          onClick={() => setIsAdding(true)}
          className="w-full py-2 bg-white/10 hover:bg-white/20 text-white text-sm rounded transition-colors"
        >
          + Add Move to History
        </button>
      )}

      {history.length > 0 && !isAdding && (
        <button
          onClick={() => onHistoryChange([])}
          className="mt-2 text-sm text-red-400 hover:text-red-300 underline"
        >
          Clear history
        </button>
      )}
    </div>
  );
}
