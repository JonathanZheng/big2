'use client';

import { useState, useCallback, useEffect, Suspense } from 'react';
import Link from 'next/link';
import { useRouter, useSearchParams } from 'next/navigation';
import { Button } from '@/components/ui/button';
import { Lobby, WaitingRoom, MultiplayerGameBoard } from '@/components/multiplayer';
import { useMultiplayerGame, useMatchmakingStatus } from '@/hooks/useMultiplayerGame';
import { PlayerIndex } from '@/lib/game/types';

type Phase = 'lobby' | 'waiting' | 'playing' | 'ended';

function PlayOnlineContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const initialGameId = searchParams.get('game');

  const [gameId, setGameId] = useState<string | null>(initialGameId);
  const [phase, setPhase] = useState<Phase>(initialGameId ? 'waiting' : 'lobby');
  const [gameResult, setGameResult] = useState<{
    isWin: boolean;
    winnerName: string;
  } | null>(null);

  const { game, error: gameError, isLoading, refresh } = useMultiplayerGame(gameId);
  const { isInQueue, matchedGameId } = useMatchmakingStatus();

  // Handle matchmaking match found
  useEffect(() => {
    if (matchedGameId && !gameId) {
      setGameId(matchedGameId);
      setPhase('waiting');
    }
  }, [matchedGameId, gameId]);

  // Update phase based on game status
  useEffect(() => {
    if (!game) return;

    if (game.status === 'waiting') {
      setPhase('waiting');
    } else if (game.status === 'in_progress') {
      setPhase('playing');
    } else if (game.status === 'completed' || game.status === 'abandoned') {
      setPhase('ended');
    }
  }, [game?.status]);

  const handleGameJoined = useCallback(
    (newGameId: string) => {
      setGameId(newGameId);
      setPhase('waiting');
      // Update URL without navigation
      window.history.pushState({}, '', `/play/online?game=${newGameId}`);
    },
    []
  );

  const handleLeaveRoom = useCallback(() => {
    setGameId(null);
    setPhase('lobby');
    setGameResult(null);
    window.history.pushState({}, '', '/play/online');
  }, []);

  const handleGameEnd = useCallback(
    (winner: PlayerIndex) => {
      if (game) {
        const isWin = winner === game.myPosition;
        const winnerName = game.players[winner]?.username || `Player ${winner}`;
        setGameResult({ isWin, winnerName });
      }
      setPhase('ended');
    },
    [game]
  );

  const handlePlayAgain = useCallback(() => {
    setGameId(null);
    setPhase('lobby');
    setGameResult(null);
    window.history.pushState({}, '', '/play/online');
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-b from-blue-900 to-blue-950 p-4 md:p-8">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <Link href="/">
            <Button variant="ghost" className="text-white hover:text-gray-300">
              ← Back to Home
            </Button>
          </Link>
          <h1 className="text-2xl font-bold text-white">Online Multiplayer</h1>
          <div className="w-24" />
        </div>

        {/* Loading state */}
        {isLoading && gameId && (
          <div className="flex items-center justify-center py-20">
            <div className="flex items-center gap-3 text-white">
              <div className="animate-spin w-6 h-6 border-2 border-white/30 border-t-white rounded-full" />
              <span>Loading game...</span>
            </div>
          </div>
        )}

        {/* Error state */}
        {gameError && (
          <div className="max-w-md mx-auto">
            <div className="bg-red-500/20 border border-red-500/30 rounded-lg p-6 text-center">
              <h2 className="text-xl font-semibold text-red-300 mb-2">Error</h2>
              <p className="text-red-200 mb-4">{gameError}</p>
              <Button onClick={handleLeaveRoom} className="bg-red-600 hover:bg-red-500">
                Back to Lobby
              </Button>
            </div>
          </div>
        )}

        {/* Lobby phase */}
        {phase === 'lobby' && !isLoading && (
          <Lobby
            onGameJoined={handleGameJoined}
            isInQueue={isInQueue}
            matchedGameId={matchedGameId}
          />
        )}

        {/* Waiting room phase */}
        {phase === 'waiting' && game && !isLoading && (
          <WaitingRoom
            gameId={game.gameId}
            roomCode={game.roomCode}
            players={game.players}
            isHost={game.isHost}
            myPosition={game.myPosition}
            onLeave={handleLeaveRoom}
          />
        )}

        {/* Playing phase */}
        {phase === 'playing' && game && !isLoading && (
          <MultiplayerGameBoard game={game} onGameEnd={handleGameEnd} />
        )}

        {/* Game ended phase */}
        {phase === 'ended' && (
          <div className="max-w-md mx-auto">
            <div className="bg-white/10 rounded-lg p-8 text-center">
              <h2 className="text-3xl font-bold text-white mb-4">
                {gameResult?.isWin ? '🎉 Victory!' : 'Game Over'}
              </h2>

              {gameResult && (
                <p className="text-blue-200 mb-6">
                  {gameResult.isWin
                    ? 'Congratulations on your win!'
                    : `${gameResult.winnerName} won the game`}
                </p>
              )}

              <div className="space-y-3">
                <Button
                  onClick={handlePlayAgain}
                  className="w-full bg-blue-600 hover:bg-blue-500"
                >
                  Play Again
                </Button>
                <Link href="/profile" className="block">
                  <Button variant="outline" className="w-full border-white/30 text-white">
                    View Profile
                  </Button>
                </Link>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function LoadingFallback() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-blue-900 to-blue-950 flex items-center justify-center">
      <div className="flex items-center gap-3 text-white">
        <div className="animate-spin w-6 h-6 border-2 border-white/30 border-t-white rounded-full" />
        <span>Loading...</span>
      </div>
    </div>
  );
}

export default function PlayOnlinePage() {
  return (
    <Suspense fallback={<LoadingFallback />}>
      <PlayOnlineContent />
    </Suspense>
  );
}
