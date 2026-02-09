'use client';

import { useState, useCallback } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import {
  createRoom,
  joinRoom,
  joinMatchmaking,
  leaveMatchmaking,
} from '@/app/play/online/actions';

interface LobbyProps {
  onGameJoined: (gameId: string) => void;
  isInQueue: boolean;
  matchedGameId: string | null;
}

export function Lobby({ onGameJoined, isInQueue, matchedGameId }: LobbyProps) {
  const [roomCode, setRoomCode] = useState('');
  const [isCreating, setIsCreating] = useState(false);
  const [isJoining, setIsJoining] = useState(false);
  const [isMatchmaking, setIsMatchmaking] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Auto-redirect when match found
  if (matchedGameId) {
    onGameJoined(matchedGameId);
  }

  const handleCreateRoom = useCallback(async () => {
    setIsCreating(true);
    setError(null);

    try {
      const result = await createRoom();
      if (result.success && result.gameId) {
        onGameJoined(result.gameId);
      } else {
        setError(result.error || 'Failed to create room');
      }
    } catch (err) {
      setError('Failed to create room');
    } finally {
      setIsCreating(false);
    }
  }, [onGameJoined]);

  const handleJoinRoom = useCallback(async () => {
    if (!roomCode.trim()) {
      setError('Please enter a room code');
      return;
    }

    setIsJoining(true);
    setError(null);

    try {
      const result = await joinRoom(roomCode.trim());
      if (result.success && result.gameId) {
        onGameJoined(result.gameId);
      } else {
        setError(result.error || 'Failed to join room');
      }
    } catch (err) {
      setError('Failed to join room');
    } finally {
      setIsJoining(false);
    }
  }, [roomCode, onGameJoined]);

  const handleMatchmaking = useCallback(async () => {
    setIsMatchmaking(true);
    setError(null);

    try {
      const result = await joinMatchmaking();
      if (result.success) {
        if (result.matched && result.gameId) {
          onGameJoined(result.gameId);
        }
        // If not matched, user is now in queue - UI will update via hook
      } else {
        setError(result.error || 'Failed to join matchmaking');
      }
    } catch (err) {
      setError('Failed to join matchmaking');
    } finally {
      setIsMatchmaking(false);
    }
  }, [onGameJoined]);

  const handleLeaveQueue = useCallback(async () => {
    try {
      await leaveMatchmaking();
    } catch (err) {
      console.error('Failed to leave queue:', err);
    }
  }, []);

  return (
    <div className="max-w-md mx-auto space-y-6">
      {/* Create Room */}
      <div className="bg-white/10 rounded-lg p-6">
        <h2 className="text-xl font-semibold text-white mb-4">Create a Room</h2>
        <p className="text-blue-200 text-sm mb-4">
          Create a private room and share the code with friends
        </p>
        <Button
          onClick={handleCreateRoom}
          disabled={isCreating || isInQueue}
          className="w-full bg-blue-600 hover:bg-blue-500"
        >
          {isCreating ? 'Creating...' : 'Create Room'}
        </Button>
      </div>

      {/* Join Room */}
      <div className="bg-white/10 rounded-lg p-6">
        <h2 className="text-xl font-semibold text-white mb-4">Join a Room</h2>
        <p className="text-blue-200 text-sm mb-4">Enter the 6-character room code to join</p>
        <div className="flex gap-2">
          <Input
            value={roomCode}
            onChange={(e) => setRoomCode(e.target.value.toUpperCase())}
            placeholder="ABCD12"
            maxLength={6}
            className="bg-white/10 border-white/20 text-white placeholder:text-white/50 uppercase"
            disabled={isInQueue}
          />
          <Button
            onClick={handleJoinRoom}
            disabled={isJoining || !roomCode.trim() || isInQueue}
            className="bg-green-600 hover:bg-green-500"
          >
            {isJoining ? 'Joining...' : 'Join'}
          </Button>
        </div>
      </div>

      {/* Matchmaking */}
      <div className="bg-white/10 rounded-lg p-6">
        <h2 className="text-xl font-semibold text-white mb-4">Quick Match</h2>
        <p className="text-blue-200 text-sm mb-4">
          Find opponents automatically based on your ELO rating
        </p>

        {isInQueue ? (
          <div className="space-y-4">
            <div className="flex items-center justify-center gap-3 p-4 bg-blue-600/20 rounded-lg">
              <div className="animate-spin w-5 h-5 border-2 border-white/30 border-t-white rounded-full" />
              <span className="text-white">Searching for opponents...</span>
            </div>
            <Button
              onClick={handleLeaveQueue}
              variant="outline"
              className="w-full border-red-400 text-red-300 hover:bg-red-600/20"
            >
              Cancel
            </Button>
          </div>
        ) : (
          <Button
            onClick={handleMatchmaking}
            disabled={isMatchmaking}
            className="w-full bg-purple-600 hover:bg-purple-500"
          >
            {isMatchmaking ? 'Finding Match...' : 'Find Match'}
          </Button>
        )}
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
