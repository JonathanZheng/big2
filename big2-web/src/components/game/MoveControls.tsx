'use client';

import { Button } from '@/components/ui/button';
import { detectMoveType, canBeat, MoveType, Move } from '@/lib/game';

interface MoveControlsProps {
  selectedCards: number[];
  lastMove: Move | null;
  isFirstMove: boolean;
  canPass: boolean;
  onPlay: () => void;
  onPass: () => void;
  onClear: () => void;
  disabled?: boolean;
}

export function MoveControls({
  selectedCards,
  lastMove,
  isFirstMove,
  canPass,
  onPlay,
  onPass,
  onClear,
  disabled = false,
}: MoveControlsProps) {
  // Check if the selected cards form a valid move
  const moveType = detectMoveType(selectedCards);
  const isValidMoveType = moveType !== MoveType.INVALID && moveType !== MoveType.PASS;

  // Check if it's a valid first move (must contain 3♦)
  const containsThreeDiamonds = selectedCards.includes(0);
  const isValidFirstMove = !isFirstMove || containsThreeDiamonds;

  // Check if the move can beat the last move
  const canBeatLastMove = lastMove === null || canBeat(selectedCards, lastMove.cards);

  // Overall validity
  const isValidPlay = selectedCards.length > 0 && isValidMoveType && isValidFirstMove && canBeatLastMove;

  // Get error message
  let errorMessage = '';
  if (selectedCards.length > 0) {
    if (!isValidMoveType) {
      errorMessage = 'Invalid combination';
    } else if (!isValidFirstMove) {
      errorMessage = 'First move must include 3♦';
    } else if (!canBeatLastMove) {
      errorMessage = 'Cannot beat last move';
    }
  }

  return (
    <div className="flex flex-col items-center gap-3">
      {errorMessage && (
        <div className="text-red-500 text-sm font-medium">
          {errorMessage}
        </div>
      )}

      <div className="flex gap-3">
        <Button
          variant="default"
          size="lg"
          onClick={onPlay}
          disabled={disabled || !isValidPlay}
          className="min-w-24"
        >
          Play
        </Button>

        <Button
          variant="outline"
          size="lg"
          onClick={onPass}
          disabled={disabled || !canPass}
          className="min-w-24"
        >
          Pass
        </Button>

        <Button
          variant="ghost"
          size="lg"
          onClick={onClear}
          disabled={disabled || selectedCards.length === 0}
          className="min-w-24"
        >
          Clear
        </Button>
      </div>
    </div>
  );
}
