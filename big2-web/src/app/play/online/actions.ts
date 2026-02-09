'use server';

import { createClient } from '@/lib/supabase/server';
import { revalidatePath } from 'next/cache';
import {
  Big2Game,
  createGame,
  createMove,
  getLegalMoves,
  PlayerIndex,
  GameState,
} from '@/lib/game';

// Generate 6-character room code
function generateRoomCode(): string {
  const chars = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789'; // Avoid confusing chars
  let code = '';
  for (let i = 0; i < 6; i++) {
    code += chars.charAt(Math.floor(Math.random() * chars.length));
  }
  return code;
}

export interface CreateRoomResult {
  success: boolean;
  gameId?: string;
  roomCode?: string;
  error?: string;
}

export async function createRoom(): Promise<CreateRoomResult> {
  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) {
    return { success: false, error: 'Not authenticated' };
  }

  const roomCode = generateRoomCode();

  const { data, error } = await supabase
    .from('games')
    .insert({
      mode: 'multiplayer',
      status: 'waiting',
      room_code: roomCode,
      player_0_id: user.id,
      state: {},
      is_private: true,
    })
    .select()
    .single();

  if (error) {
    console.error('Failed to create room:', error);
    return { success: false, error: error.message };
  }

  return { success: true, gameId: data.id, roomCode };
}

export interface JoinRoomResult {
  success: boolean;
  gameId?: string;
  position?: number;
  error?: string;
}

export async function joinRoom(roomCode: string): Promise<JoinRoomResult> {
  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) {
    return { success: false, error: 'Not authenticated' };
  }

  // Find game by room code
  const { data: game, error } = await supabase
    .from('games')
    .select('*')
    .eq('room_code', roomCode.toUpperCase())
    .eq('status', 'waiting')
    .single();

  if (error || !game) {
    return { success: false, error: 'Room not found or game already started' };
  }

  // Check if user is already in the game
  const slots = [game.player_0_id, game.player_1_id, game.player_2_id, game.player_3_id];
  const existingPosition = slots.indexOf(user.id);
  if (existingPosition !== -1) {
    return { success: true, gameId: game.id, position: existingPosition };
  }

  // Find empty slot
  const emptySlot = slots.findIndex((s) => s === null);

  if (emptySlot === -1) {
    return { success: false, error: 'Room is full' };
  }

  // Use atomic join function
  const { data: joinSuccess, error: joinError } = await supabase.rpc('join_game_room', {
    p_game_id: game.id,
    p_user_id: user.id,
    p_position: emptySlot,
  });

  if (joinError || !joinSuccess) {
    return { success: false, error: 'Failed to join room - slot may have been taken' };
  }

  return { success: true, gameId: game.id, position: emptySlot };
}

export interface MatchmakingResult {
  success: boolean;
  matched?: boolean;
  gameId?: string;
  error?: string;
}

export async function joinMatchmaking(): Promise<MatchmakingResult> {
  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) {
    return { success: false, error: 'Not authenticated' };
  }

  // Get user's ELO
  const { data: stats } = await supabase
    .from('player_stats')
    .select('elo_rating')
    .eq('user_id', user.id)
    .single();

  const elo = stats?.elo_rating ?? 1000;

  // Check if already in queue
  const { data: existingEntry } = await supabase
    .from('matchmaking_queue')
    .select('*')
    .eq('user_id', user.id)
    .eq('status', 'waiting')
    .single();

  if (!existingEntry) {
    // Add to queue
    const { error: insertError } = await supabase
      .from('matchmaking_queue')
      .insert({ user_id: user.id, elo_rating: elo });

    if (insertError) {
      return { success: false, error: 'Failed to join queue' };
    }
  }

  // Try to find a match
  const { data: gameId, error: matchError } = await supabase.rpc('find_match', {
    p_user_id: user.id,
    p_elo: elo,
  });

  if (matchError) {
    console.error('Match error:', matchError);
  }

  if (gameId) {
    return { success: true, matched: true, gameId };
  }

  return { success: true, matched: false };
}

export async function leaveMatchmaking(): Promise<{ success: boolean; error?: string }> {
  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) {
    return { success: false, error: 'Not authenticated' };
  }

  const { error } = await supabase.rpc('leave_matchmaking', {
    p_user_id: user.id,
  });

  if (error) {
    return { success: false, error: error.message };
  }

  return { success: true };
}

export async function leaveRoom(gameId: string): Promise<{ success: boolean; error?: string }> {
  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) {
    return { success: false, error: 'Not authenticated' };
  }

  // Get current game
  const { data: game } = await supabase.from('games').select('*').eq('id', gameId).single();

  if (!game || game.status !== 'waiting') {
    return { success: false, error: 'Cannot leave - game not found or already started' };
  }

  // Find which slot the user is in
  const slots = [game.player_0_id, game.player_1_id, game.player_2_id, game.player_3_id];
  const position = slots.indexOf(user.id);

  if (position === -1) {
    return { success: false, error: 'Not in this game' };
  }

  // If host (position 0) leaves and game hasn't started, delete the game
  if (position === 0) {
    await supabase.from('games').delete().eq('id', gameId);
  } else {
    // Remove player from slot
    const updateObj: Record<string, null> = {};
    updateObj[`player_${position}_id`] = null;
    await supabase.from('games').update(updateObj).eq('id', gameId);
  }

  return { success: true };
}

export interface StartGameResult {
  success: boolean;
  error?: string;
}

export async function startGame(gameId: string): Promise<StartGameResult> {
  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) {
    return { success: false, error: 'Not authenticated' };
  }

  // Get current game
  const { data: game } = await supabase.from('games').select('*').eq('id', gameId).single();

  if (!game) {
    return { success: false, error: 'Game not found' };
  }

  // Only host (player_0) can start the game
  if (game.player_0_id !== user.id) {
    return { success: false, error: 'Only the host can start the game' };
  }

  // Create a new game and get initial state
  const newGame = createGame();
  const initialState = newGame.getState();

  // Start the game using database function
  const { data: success, error } = await supabase.rpc('start_game', {
    p_game_id: gameId,
    p_initial_state: initialState,
  });

  if (error || !success) {
    return { success: false, error: 'Failed to start game - ensure all 4 players have joined' };
  }

  return { success: true };
}

export interface MakeMoveResult {
  success: boolean;
  done?: boolean;
  winner?: number;
  error?: string;
}

export async function makeMove(gameId: string, cards: number[]): Promise<MakeMoveResult> {
  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) {
    return { success: false, error: 'Not authenticated' };
  }

  // Get current game state
  const { data: game, error } = await supabase.from('games').select('*').eq('id', gameId).single();

  if (error || !game) {
    return { success: false, error: 'Game not found' };
  }

  if (game.status !== 'in_progress') {
    return { success: false, error: 'Game is not in progress' };
  }

  // Determine player position
  const slots = [game.player_0_id, game.player_1_id, game.player_2_id, game.player_3_id];
  const playerPositionIndex = slots.indexOf(user.id);

  if (playerPositionIndex === -1) {
    return { success: false, error: 'Not a player in this game' };
  }

  const playerPosition = playerPositionIndex as PlayerIndex;

  // Reconstruct game from state
  const gameState = game.state as GameState;

  if (gameState.currentPlayer !== playerPosition) {
    return { success: false, error: 'Not your turn' };
  }

  // Validate move using a temporary game instance
  const tempGame = createGame();
  // Manually set the state (this is a workaround since Big2Game doesn't expose setState)
  Object.assign((tempGame as unknown as { state: GameState }).state, gameState);

  const legalMoves = getLegalMoves(tempGame, playerPosition);
  const sortedCards = [...cards].sort((a, b) => a - b);
  const isLegalMove = legalMoves.some(
    (m) =>
      m.cards.length === sortedCards.length &&
      [...m.cards].sort((a, b) => a - b).every((c, i) => c === sortedCards[i])
  );

  if (!isLegalMove) {
    return { success: false, error: 'Invalid move' };
  }

  // Execute move
  const move = createMove(sortedCards, playerPosition);
  tempGame.step(move);

  const newState = tempGame.getState();
  const updateData: {
    state: GameState;
    turn_started_at: string;
    status?: string;
    winner_position?: number;
  } = {
    state: newState,
    turn_started_at: new Date().toISOString(),
  };

  if (tempGame.isDone()) {
    updateData.status = 'completed';
    updateData.winner_position = tempGame.getWinner()!;
  }

  // Update game state
  const { error: updateError } = await supabase.from('games').update(updateData).eq('id', gameId);

  if (updateError) {
    return { success: false, error: 'Failed to update game state' };
  }

  // If game is done, update stats for all players
  if (tempGame.isDone()) {
    await updateMultiplayerStats(gameId, tempGame.getWinner()!, slots);
  }

  return {
    success: true,
    done: tempGame.isDone(),
    winner: tempGame.getWinner() ?? undefined,
  };
}

// Update stats for multiplayer game
async function updateMultiplayerStats(
  gameId: string,
  winnerPosition: PlayerIndex,
  playerIds: (string | null)[]
): Promise<void> {
  const supabase = await createClient();
  const K_FACTOR = 32;

  // Get all player stats
  const validPlayerIds = playerIds.filter((id): id is string => id !== null);

  const { data: allStats } = await supabase
    .from('player_stats')
    .select('*')
    .in('user_id', validPlayerIds);

  if (!allStats) return;

  // Calculate average opponent ELO for each player
  const playerElos = new Map<string, number>();
  for (const stats of allStats) {
    playerElos.set(stats.user_id, stats.elo_rating);
  }

  // Update each player's stats
  for (let i = 0; i < 4; i++) {
    const playerId = playerIds[i];
    if (!playerId) continue;

    const stats = allStats.find((s) => s.user_id === playerId);
    if (!stats) continue;

    const isWin = i === winnerPosition;
    const currentElo = stats.elo_rating;

    // Calculate average opponent ELO
    let opponentEloSum = 0;
    let opponentCount = 0;
    for (let j = 0; j < 4; j++) {
      if (j !== i && playerIds[j]) {
        const oppElo = playerElos.get(playerIds[j]!) ?? 1000;
        opponentEloSum += oppElo;
        opponentCount++;
      }
    }
    const avgOpponentElo = opponentCount > 0 ? opponentEloSum / opponentCount : 1000;

    // Calculate new ELO
    const expectedScore = 1 / (1 + Math.pow(10, (avgOpponentElo - currentElo) / 400));
    const actualScore = isWin ? 1 : 0;
    const newElo = Math.round(currentElo + K_FACTOR * (actualScore - expectedScore));

    const newStreak = isWin ? stats.win_streak + 1 : 0;
    const newBestStreak = Math.max(stats.best_win_streak, newStreak);
    const newHighestElo = Math.max(stats.highest_elo, newElo);

    // Update stats
    await supabase
      .from('player_stats')
      .update({
        games_played: stats.games_played + 1,
        games_won: stats.games_won + (isWin ? 1 : 0),
        elo_rating: newElo,
        highest_elo: newHighestElo,
        win_streak: newStreak,
        best_win_streak: newBestStreak,
      })
      .eq('user_id', playerId);
  }
}

export interface GameInfo {
  id: string;
  roomCode: string | null;
  status: string;
  players: {
    position: number;
    id: string | null;
    username: string | null;
  }[];
  isHost: boolean;
  myPosition: number | null;
}

export async function getGameInfo(gameId: string): Promise<{ success: boolean; game?: GameInfo; error?: string }> {
  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) {
    return { success: false, error: 'Not authenticated' };
  }

  const { data: game, error } = await supabase
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

  if (error || !game) {
    return { success: false, error: 'Game not found' };
  }

  const slots = [game.player_0_id, game.player_1_id, game.player_2_id, game.player_3_id];
  const myPosition = slots.indexOf(user.id);

  const players = [
    { position: 0, id: game.player_0_id, username: (game.player_0 as { username: string } | null)?.username ?? null },
    { position: 1, id: game.player_1_id, username: (game.player_1 as { username: string } | null)?.username ?? null },
    { position: 2, id: game.player_2_id, username: (game.player_2 as { username: string } | null)?.username ?? null },
    { position: 3, id: game.player_3_id, username: (game.player_3 as { username: string } | null)?.username ?? null },
  ];

  return {
    success: true,
    game: {
      id: game.id,
      roomCode: game.room_code,
      status: game.status,
      players,
      isHost: game.player_0_id === user.id,
      myPosition: myPosition === -1 ? null : myPosition,
    },
  };
}
