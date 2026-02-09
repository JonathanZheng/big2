'use client';

import { useState, useCallback, useEffect, useRef } from 'react';
import { cn } from '@/lib/utils';
import { PlayerHand } from '@/components/game/PlayerHand';
import { PlayArea, SelectedCardsPreview } from '@/components/game/PlayArea';
import { MoveControls } from '@/components/game/MoveControls';
import { MultiplayerGameState, MultiplayerPlayer } from '@/hooks/useMultiplayerGame';
import { makeMove } from '@/app/play/online/actions';
import {
  createMove,
  getLegalMoves,
  Big2Game,
  createGame,
  PlayerIndex,
  GameState,
} from '@/lib/game';

interface MultiplayerGameBoardProps {
  game: MultiplayerGameState;
  onGameEnd?: (winner: PlayerIndex) => void;
}

export function MultiplayerGameBoard({ game, onGameEnd }: MultiplayerGameBoardProps) {
  const [selectedCards, setSelectedCards] = useState<number[]>([]);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [timeLeft, setTimeLeft] = useState<number>(game.turnTimeoutSeconds);
  const timerRef = useRef<NodeJS.Timeout | null>(null);

  const { state, myPosition, isMyTurn, players, status, turnStartedAt, turnTimeoutSeconds } = game;

  // Reset selection when turn changes
  useEffect(() => {
    setSelectedCards([]);
    setError(null);
  }, [state?.currentPlayer]);

  // Turn timer
  useEffect(() => {
    if (!turnStartedAt || status !== 'in_progress') {
      setTimeLeft(turnTimeoutSeconds);
      return;
    }

    const updateTimer = () => {
      const elapsed = (Date.now() - turnStartedAt.getTime()) / 1000;
      const remaining = Math.max(0, turnTimeoutSeconds - elapsed);
      setTimeLeft(Math.ceil(remaining));
    };

    updateTimer();
    timerRef.current = setInterval(updateTimer, 1000);

    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
    };
  }, [turnStartedAt, turnTimeoutSeconds, status]);

  // Notify parent when game ends
  useEffect(() => {
    if (status === 'completed' && state?.winner !== null && state?.winner !== undefined) {
      onGameEnd?.(state.winner);
    }
  }, [status, state?.winner, onGameEnd]);

  const handleCardClick = useCallback(
    (card: number) => {
      if (!isMyTurn || isSubmitting) return;

      setSelectedCards((prev) =>
        prev.includes(card) ? prev.filter((c) => c !== card) : [...prev, card]
      );
    },
    [isMyTurn, isSubmitting]
  );

  const handlePlay = useCallback(async () => {
    if (!isMyTurn || isSubmitting || myPosition === null) return;

    setIsSubmitting(true);
    setError(null);

    try {
      const result = await makeMove(game.gameId, selectedCards);
      if (!result.success) {
        setError(result.error || 'Failed to play');
      } else {
        setSelectedCards([]);
      }
    } catch (err) {
      setError('Failed to play');
    } finally {
      setIsSubmitting(false);
    }
  }, [game.gameId, selectedCards, isMyTurn, isSubmitting, myPosition]);

  const handlePass = useCallback(async () => {
    if (!isMyTurn || isSubmitting || myPosition === null) return;

    setIsSubmitting(true);
    setError(null);

    try {
      const result = await makeMove(game.gameId, []);
      if (!result.success) {
        setError(result.error || 'Failed to pass');
      }
    } catch (err) {
      setError('Failed to pass');
    } finally {
      setIsSubmitting(false);
    }
  }, [game.gameId, isMyTurn, isSubmitting, myPosition]);

  const handleClear = useCallback(() => {
    setSelectedCards([]);
  }, []);

  // Get player info relative to current player's position
  const getRelativePlayer = (relativePosition: number): MultiplayerPlayer => {
    if (myPosition === null) {
      return players[relativePosition];
    }
    const actualPosition = (myPosition + relativePosition) % 4;
    return players[actualPosition];
  };

  if (!state) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-gray-500">Waiting for game state...</div>
      </div>
    );
  }

  const currentPlayer = state.currentPlayer;
  const lastMove = state.lastMove;
  const canPass = lastMove !== null && !state.firstMove;

  // Get player hands (hide opponents' cards)
  const getHandForDisplay = (position: number): number[] => {
    if (position === myPosition) {
      return state.hands[position] || [];
    }
    // For opponents, return array of length equal to their hand size
    const handSize = state.hands[position]?.length || 0;
    return Array(handSize).fill(-1);
  };

  // Relative positions: 0 = bottom (me), 1 = left, 2 = top, 3 = right
  const bottomPlayer = getRelativePlayer(0);
  const leftPlayer = getRelativePlayer(1);
  const topPlayer = getRelativePlayer(2);
  const rightPlayer = getRelativePlayer(3);

  const isFirstMove = state.firstMove;

  return (
    <div className="relative w-full h-full min-h-[700px] bg-green-800 rounded-xl p-4 flex flex-col">
      {/* Game end overlay */}
      {status === 'completed' && (
        <div className="absolute inset-0 bg-black/60 flex items-center justify-center z-20 rounded-xl">
          <div className="bg-white p-8 rounded-xl shadow-2xl text-center">
            <h2 className="text-3xl font-bold mb-4">
              {state.winner === myPosition ? '🎉 You Won!' : 'Game Over'}
            </h2>
            <p className="text-gray-600 mb-6">
              {state.winner === myPosition
                ? 'Congratulations!'
                : `${players[state.winner!]?.username || `Player ${state.winner}`} wins!`}
            </p>
          </div>
        </div>
      )}

      {/* Turn timer */}
      {status === 'in_progress' && (
        <div className="absolute top-4 right-4 z-10">
          <div
            className={cn(
              'px-3 py-1 rounded-full font-mono text-sm',
              timeLeft <= 10
                ? 'bg-red-500/80 text-white animate-pulse'
                : 'bg-white/20 text-white'
            )}
          >
            {timeLeft}s
          </div>
        </div>
      )}

      {/* Error display */}
      {error && (
        <div className="absolute top-4 left-1/2 transform -translate-x-1/2 z-10 bg-red-500/80 text-white px-4 py-2 rounded-lg">
          {error}
        </div>
      )}

      {/* Top player */}
      <div className="flex justify-center mb-4">
        <PlayerHand
          cards={getHandForDisplay(topPlayer.position)}
          selectedCards={[]}
          position="top"
          playerName={topPlayer.username || `Player ${topPlayer.position}`}
          showCards={false}
          isCurrentPlayer={currentPlayer === topPlayer.position}
        />
      </div>

      {/* Middle section */}
      <div className="flex flex-1 items-center justify-between px-4">
        {/* Left player */}
        <div className="flex items-center">
          <PlayerHand
            cards={getHandForDisplay(leftPlayer.position)}
            selectedCards={[]}
            position="left"
            playerName={leftPlayer.username || `Player ${leftPlayer.position}`}
            showCards={false}
            isCurrentPlayer={currentPlayer === leftPlayer.position}
          />
        </div>

        {/* Center - Play Area */}
        <div className="flex-1 flex flex-col items-center justify-center">
          <div className="bg-green-700/50 rounded-xl p-6 min-w-[300px]">
            <PlayArea
              lastMove={lastMove}
              message={isFirstMove ? 'First move - must include 3♦' : undefined}
            />
          </div>

          {/* Turn indicator */}
          <div className="mt-4 text-white/80 text-sm">
            {isSubmitting ? (
              <span className="flex items-center gap-2">
                <span className="animate-spin">⚙️</span>
                Submitting move...
              </span>
            ) : isMyTurn ? (
              <span className="text-yellow-300 font-medium">Your turn!</span>
            ) : (
              <span>
                Waiting for {players[currentPlayer]?.username || `Player ${currentPlayer}`}...
              </span>
            )}
          </div>
        </div>

        {/* Right player */}
        <div className="flex items-center">
          <PlayerHand
            cards={getHandForDisplay(rightPlayer.position)}
            selectedCards={[]}
            position="right"
            playerName={rightPlayer.username || `Player ${rightPlayer.position}`}
            showCards={false}
            isCurrentPlayer={currentPlayer === rightPlayer.position}
          />
        </div>
      </div>

      {/* Bottom section - Current player */}
      <div className="mt-4 flex flex-col items-center gap-4">
        {/* Selected cards preview */}
        {isMyTurn && myPosition !== null && (
          <div className="bg-green-700/30 rounded-lg px-6 py-3">
            <SelectedCardsPreview cards={selectedCards} />
          </div>
        )}

        {/* Controls */}
        {isMyTurn && myPosition !== null && (
          <MoveControls
            selectedCards={selectedCards}
            lastMove={lastMove}
            isFirstMove={isFirstMove}
            canPass={canPass}
            onPlay={handlePlay}
            onPass={handlePass}
            onClear={handleClear}
            disabled={status !== 'in_progress' || isSubmitting}
          />
        )}

        {/* Player's hand */}
        {myPosition !== null && (
          <PlayerHand
            cards={state.hands[myPosition] || []}
            selectedCards={selectedCards}
            onCardClick={handleCardClick}
            position="bottom"
            playerName={bottomPlayer.username || 'You'}
            showCards={true}
            isCurrentPlayer={isMyTurn}
            disabled={!isMyTurn || status !== 'in_progress' || isSubmitting}
          />
        )}
      </div>
    </div>
  );
}
