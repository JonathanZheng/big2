'use server';

import { createClient } from '@/lib/supabase/server';
import { PlayerIndex } from '@/lib/game/types';

const K_FACTOR = 32; // ELO adjustment factor

function calculateNewElo(currentElo: number, opponentElo: number, won: boolean): number {
  const expectedScore = 1 / (1 + Math.pow(10, (opponentElo - currentElo) / 400));
  const actualScore = won ? 1 : 0;
  return Math.round(currentElo + K_FACTOR * (actualScore - expectedScore));
}

export interface GameResultResponse {
  success: boolean;
  error?: string;
  isWin?: boolean;
  newElo?: number;
  eloChange?: number;
  newStreak?: number;
}

export async function recordGameResult(
  winner: PlayerIndex,
  humanPosition: PlayerIndex
): Promise<GameResultResponse> {
  const supabase = await createClient();
  const { data: { user } } = await supabase.auth.getUser();

  if (!user) {
    console.error('User not authenticated - cannot record stats');
    return { success: false, error: 'Not authenticated' };
  }

  const isWin = winner === humanPosition;

  // Get current stats
  const { data: stats, error: statsError } = await supabase
    .from('player_stats')
    .select('*')
    .eq('user_id', user.id)
    .single();

  if (statsError) {
    console.error('Failed to fetch stats:', statsError);
    return { success: false, error: statsError.message };
  }

  // Calculate new values
  const botElo = 1000; // Bots are rated at 1000
  const newElo = calculateNewElo(stats.elo_rating, botElo, isWin);
  const newStreak = isWin ? stats.win_streak + 1 : 0;
  const newBestStreak = Math.max(stats.best_win_streak, newStreak);
  const newHighestElo = Math.max(stats.highest_elo, newElo);

  // Update stats
  const { error: updateError } = await supabase
    .from('player_stats')
    .update({
      games_played: stats.games_played + 1,
      games_won: stats.games_won + (isWin ? 1 : 0),
      elo_rating: newElo,
      highest_elo: newHighestElo,
      win_streak: newStreak,
      best_win_streak: newBestStreak,
    })
    .eq('user_id', user.id);

  if (updateError) {
    console.error('Failed to update stats:', updateError);
    return { success: false, error: updateError.message };
  }

  return {
    success: true,
    isWin,
    newElo,
    eloChange: newElo - stats.elo_rating,
    newStreak,
  };
}
