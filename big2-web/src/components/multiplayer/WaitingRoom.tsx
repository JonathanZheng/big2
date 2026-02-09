'use client';

import { useState, useCallback } from 'react';
import { Button } from '@/components/ui/button';
import { MultiplayerPlayer } from '@/hooks/useMultiplayerGame';
import { startGame, leaveRoom } from '@/app/play/online/actions';

interface WaitingRoomProps {
  gameId: string;
  roomCode: string | null;
  players: MultiplayerPlayer[];
  isHost: boolean;
  myPosition: number | null;
  onLeave: () => void;
}

export function WaitingRoom({
  gameId,
  roomCode,
  players,
  isHost,
  myPosition,
  onLeave,
}: WaitingRoomProps) {
  const [isStarting, setIsStarting] = useState(false);
  const [isLeaving, setIsLeaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const filledSlots = players.filter((p) => p.id !== null).length;
  const canStart = filledSlots === 4;

  const handleStart = useCallback(async () => {
    setIsStarting(true);
    setError(null);

    try {
      const result = await startGame(gameId);
      if (!result.success) {
        setError(result.error || 'Failed to start game');
      }
      // If successful, the game status will update via real-time subscription
    } catch (err) {
      setError('Failed to start game');
    } finally {
      setIsStarting(false);
    }
  }, [gameId]);

  const handleLeave = useCallback(async () => {
    setIsLeaving(true);

    try {
      await leaveRoom(gameId);
      onLeave();
    } catch (err) {
      console.error('Failed to leave room:', err);
    } finally {
      setIsLeaving(false);
    }
  }, [gameId, onLeave]);

  const copyRoomCode = useCallback(() => {
    if (roomCode) {
      navigator.clipboard.writeText(roomCode);
    }
  }, [roomCode]);

  return (
    <div className="max-w-md mx-auto space-y-6">
      {/* Room Code Display */}
      {roomCode && (
        <div className="bg-white/10 rounded-lg p-6 text-center">
          <h2 className="text-lg text-blue-200 mb-2">Room Code</h2>
          <div
            className="text-4xl font-mono font-bold text-white tracking-widest cursor-pointer hover:text-blue-300 transition-colors"
            onClick={copyRoomCode}
            title="Click to copy"
          >
            {roomCode}
          </div>
          <p className="text-blue-300 text-sm mt-2">Share this code with friends to join</p>
        </div>
      )}

      {/* Player Slots */}
      <div className="bg-white/10 rounded-lg p-6">
        <h2 className="text-xl font-semibold text-white mb-4">
          Players ({filledSlots}/4)
        </h2>

        <div className="space-y-3">
          {players.map((player) => (
            <div
              key={player.position}
              className={`flex items-center justify-between p-3 rounded-lg ${
                player.id
                  ? player.position === myPosition
                    ? 'bg-blue-600/30 border border-blue-500/50'
                    : 'bg-white/10'
                  : 'bg-white/5 border border-dashed border-white/20'
              }`}
            >
              <div className="flex items-center gap-3">
                <div
                  className={`w-10 h-10 rounded-full flex items-center justify-center text-lg font-bold ${
                    player.id ? 'bg-blue-500 text-white' : 'bg-white/10 text-white/30'
                  }`}
                >
                  {player.id ? player.username?.charAt(0).toUpperCase() || 'P' : '?'}
                </div>
                <div>
                  <div className={player.id ? 'text-white' : 'text-white/30'}>
                    {player.id
                      ? player.username || `Player ${player.position}`
                      : 'Waiting for player...'}
                  </div>
                  <div className="text-xs text-white/50">
                    Position {player.position}
                    {player.position === 0 && ' (Host)'}
                    {player.position === myPosition && ' (You)'}
                  </div>
                </div>
              </div>

              {player.id && (
                <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse" title="Connected" />
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Actions */}
      <div className="space-y-3">
        {isHost ? (
          <Button
            onClick={handleStart}
            disabled={!canStart || isStarting}
            className={`w-full py-6 text-lg ${
              canStart
                ? 'bg-green-600 hover:bg-green-500'
                : 'bg-gray-600 cursor-not-allowed'
            }`}
          >
            {isStarting
              ? 'Starting...'
              : canStart
              ? 'Start Game'
              : `Waiting for ${4 - filledSlots} more player${4 - filledSlots > 1 ? 's' : ''}`}
          </Button>
        ) : (
          <div className="text-center py-4 text-blue-200">
            <div className="flex items-center justify-center gap-2">
              <div className="animate-spin w-4 h-4 border-2 border-blue-300/30 border-t-blue-300 rounded-full" />
              Waiting for host to start the game...
            </div>
          </div>
        )}

        <Button
          onClick={handleLeave}
          disabled={isLeaving}
          variant="outline"
          className="w-full border-red-400/50 text-red-300 hover:bg-red-600/20"
        >
          {isLeaving ? 'Leaving...' : 'Leave Room'}
        </Button>
      </div>

      {/* Error display */}
      {error && (
        <div className="bg-red-500/20 border border-red-500/30 rounded-lg p-4 text-red-200">
          {error}
        </div>
      )}
    </div>
  );
}
