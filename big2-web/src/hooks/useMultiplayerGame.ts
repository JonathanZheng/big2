'use client';

import { useEffect, useState, useCallback, useRef } from 'react';
import { createClient } from '@/lib/supabase/client';
import { GameState, PlayerIndex } from '@/lib/game/types';
import { RealtimeChannel } from '@supabase/supabase-js';

export interface MultiplayerPlayer {
  id: string | null;
  username: string | null;
  position: number;
}

export interface MultiplayerGameState {
  gameId: string;
  roomCode: string | null;
  state: GameState | null;
  players: MultiplayerPlayer[];
  myPosition: PlayerIndex | null;
  status: 'waiting' | 'in_progress' | 'completed' | 'abandoned';
  isMyTurn: boolean;
  isHost: boolean;
  turnStartedAt: Date | null;
  turnTimeoutSeconds: number;
}

interface UseMultiplayerGameResult {
  game: MultiplayerGameState | null;
  error: string | null;
  isLoading: boolean;
  refresh: () => Promise<void>;
}

export function useMultiplayerGame(gameId: string | null): UseMultiplayerGameResult {
  const [game, setGame] = useState<MultiplayerGameState | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const channelRef = useRef<RealtimeChannel | null>(null);
  const supabase = createClient();

  // Load game data
  const loadGame = useCallback(async () => {
    if (!gameId) {
      setGame(null);
      setIsLoading(false);
      return;
    }

    try {
      const {
        data: { user },
      } = await supabase.auth.getUser();

      if (!user) {
        setError('Not authenticated');
        setIsLoading(false);
        return;
      }

      const { data, error: fetchError } = await supabase
        .from('games')
        .select(
          `
          *,
          player_0:profiles!games_player_0_id_fkey(id, username),
          player_1:profiles!games_player_1_id_fkey(id, username),
          player_2:profiles!games_player_2_id_fkey(id, username),
          player_3:profiles!games_player_3_id_fkey(id, username)
        `
        )
        .eq('id', gameId)
        .single();

      if (fetchError) {
        setError(fetchError.message);
        setIsLoading(false);
        return;
      }

      const slots = [data.player_0_id, data.player_1_id, data.player_2_id, data.player_3_id];
      const myPosition = slots.indexOf(user.id);

      const players: MultiplayerPlayer[] = [
        {
          position: 0,
          id: data.player_0_id,
          username: (data.player_0 as { username: string } | null)?.username ?? null,
        },
        {
          position: 1,
          id: data.player_1_id,
          username: (data.player_1 as { username: string } | null)?.username ?? null,
        },
        {
          position: 2,
          id: data.player_2_id,
          username: (data.player_2 as { username: string } | null)?.username ?? null,
        },
        {
          position: 3,
          id: data.player_3_id,
          username: (data.player_3 as { username: string } | null)?.username ?? null,
        },
      ];

      const gameState = data.state as GameState | null;
      const isMyTurn =
        gameState && myPosition !== -1 ? gameState.currentPlayer === myPosition : false;

      setGame({
        gameId: data.id,
        roomCode: data.room_code,
        state: gameState && Object.keys(gameState).length > 0 ? gameState : null,
        players,
        myPosition: myPosition === -1 ? null : (myPosition as PlayerIndex),
        status: data.status as 'waiting' | 'in_progress' | 'completed' | 'abandoned',
        isMyTurn,
        isHost: data.player_0_id === user.id,
        turnStartedAt: data.turn_started_at ? new Date(data.turn_started_at) : null,
        turnTimeoutSeconds: data.turn_timeout_seconds || 30,
      });

      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load game');
    } finally {
      setIsLoading(false);
    }
  }, [gameId, supabase]);

  // Subscribe to game changes
  useEffect(() => {
    if (!gameId) {
      setGame(null);
      setIsLoading(false);
      return;
    }

    loadGame();

    // Set up real-time subscription
    const channel = supabase
      .channel(`game:${gameId}`)
      .on(
        'postgres_changes',
        {
          event: 'UPDATE',
          schema: 'public',
          table: 'games',
          filter: `id=eq.${gameId}`,
        },
        (payload) => {
          // Update game state from real-time payload
          const newData = payload.new as Record<string, unknown>;

          setGame((prev) => {
            if (!prev) return null;

            const gameState = newData.state as GameState | null;
            const isMyTurn =
              gameState && prev.myPosition !== null
                ? gameState.currentPlayer === prev.myPosition
                : false;

            return {
              ...prev,
              state: gameState && Object.keys(gameState).length > 0 ? gameState : null,
              status: newData.status as 'waiting' | 'in_progress' | 'completed' | 'abandoned',
              isMyTurn,
              turnStartedAt: newData.turn_started_at
                ? new Date(newData.turn_started_at as string)
                : null,
            };
          });

          // Refresh full data to get updated player list
          loadGame();
        }
      )
      .subscribe();

    channelRef.current = channel;

    return () => {
      if (channelRef.current) {
        supabase.removeChannel(channelRef.current);
        channelRef.current = null;
      }
    };
  }, [gameId, supabase, loadGame]);

  return {
    game,
    error,
    isLoading,
    refresh: loadGame,
  };
}

// Hook for matchmaking queue status
export function useMatchmakingStatus() {
  const [isInQueue, setIsInQueue] = useState(false);
  const [matchedGameId, setMatchedGameId] = useState<string | null>(null);
  const supabase = createClient();
  const channelRef = useRef<RealtimeChannel | null>(null);

  useEffect(() => {
    const checkQueue = async () => {
      const {
        data: { user },
      } = await supabase.auth.getUser();

      if (!user) return;

      const { data } = await supabase
        .from('matchmaking_queue')
        .select('*')
        .eq('user_id', user.id)
        .single();

      if (data) {
        setIsInQueue(data.status === 'waiting');
        if (data.status === 'matched') {
          // Find the game
          const { data: games } = await supabase
            .from('games')
            .select('id')
            .or(
              `player_0_id.eq.${user.id},player_1_id.eq.${user.id},player_2_id.eq.${user.id},player_3_id.eq.${user.id}`
            )
            .eq('mode', 'multiplayer')
            .eq('status', 'waiting')
            .order('created_at', { ascending: false })
            .limit(1);

          if (games && games.length > 0) {
            setMatchedGameId(games[0].id);
          }
        }
      }
    };

    checkQueue();

    // Subscribe to queue changes
    const setupSubscription = async () => {
      const {
        data: { user },
      } = await supabase.auth.getUser();

      if (!user) return;

      const channel = supabase
        .channel('matchmaking')
        .on(
          'postgres_changes',
          {
            event: '*',
            schema: 'public',
            table: 'matchmaking_queue',
            filter: `user_id=eq.${user.id}`,
          },
          () => {
            checkQueue();
          }
        )
        .subscribe();

      channelRef.current = channel;
    };

    setupSubscription();

    return () => {
      if (channelRef.current) {
        supabase.removeChannel(channelRef.current);
      }
    };
  }, [supabase]);

  return { isInQueue, matchedGameId };
}
