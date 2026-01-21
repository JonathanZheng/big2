export type Json =
  | string
  | number
  | boolean
  | null
  | { [key: string]: Json | undefined }
  | Json[];

export interface Database {
  public: {
    Tables: {
      profiles: {
        Row: {
          id: string;
          username: string;
          display_name: string | null;
          avatar_url: string | null;
          created_at: string;
          updated_at: string;
        };
        Insert: {
          id: string;
          username: string;
          display_name?: string | null;
          avatar_url?: string | null;
          created_at?: string;
          updated_at?: string;
        };
        Update: {
          id?: string;
          username?: string;
          display_name?: string | null;
          avatar_url?: string | null;
          created_at?: string;
          updated_at?: string;
        };
      };
      player_stats: {
        Row: {
          user_id: string;
          games_played: number;
          games_won: number;
          elo_rating: number;
          highest_elo: number;
          win_streak: number;
          best_win_streak: number;
          updated_at: string;
        };
        Insert: {
          user_id: string;
          games_played?: number;
          games_won?: number;
          elo_rating?: number;
          highest_elo?: number;
          win_streak?: number;
          best_win_streak?: number;
          updated_at?: string;
        };
        Update: {
          user_id?: string;
          games_played?: number;
          games_won?: number;
          elo_rating?: number;
          highest_elo?: number;
          win_streak?: number;
          best_win_streak?: number;
          updated_at?: string;
        };
      };
      games: {
        Row: {
          id: string;
          mode: 'bot' | 'multiplayer' | 'test';
          status: 'waiting' | 'in_progress' | 'completed' | 'abandoned';
          room_code: string | null;
          player_0_id: string | null;
          player_1_id: string | null;
          player_2_id: string | null;
          player_3_id: string | null;
          bot_difficulties: Json | null;
          state: Json;
          winner_position: number | null;
          created_at: string;
          updated_at: string;
        };
        Insert: {
          id?: string;
          mode: 'bot' | 'multiplayer' | 'test';
          status?: 'waiting' | 'in_progress' | 'completed' | 'abandoned';
          room_code?: string | null;
          player_0_id?: string | null;
          player_1_id?: string | null;
          player_2_id?: string | null;
          player_3_id?: string | null;
          bot_difficulties?: Json | null;
          state: Json;
          winner_position?: number | null;
          created_at?: string;
          updated_at?: string;
        };
        Update: {
          id?: string;
          mode?: 'bot' | 'multiplayer' | 'test';
          status?: 'waiting' | 'in_progress' | 'completed' | 'abandoned';
          room_code?: string | null;
          player_0_id?: string | null;
          player_1_id?: string | null;
          player_2_id?: string | null;
          player_3_id?: string | null;
          bot_difficulties?: Json | null;
          state?: Json;
          winner_position?: number | null;
          created_at?: string;
          updated_at?: string;
        };
      };
      matchmaking_queue: {
        Row: {
          id: string;
          user_id: string;
          elo_rating: number;
          joined_at: string;
          status: 'waiting' | 'matched' | 'cancelled';
        };
        Insert: {
          id?: string;
          user_id: string;
          elo_rating: number;
          joined_at?: string;
          status?: 'waiting' | 'matched' | 'cancelled';
        };
        Update: {
          id?: string;
          user_id?: string;
          elo_rating?: number;
          joined_at?: string;
          status?: 'waiting' | 'matched' | 'cancelled';
        };
      };
    };
  };
}

// Convenience types
export type Profile = Database['public']['Tables']['profiles']['Row'];
export type PlayerStats = Database['public']['Tables']['player_stats']['Row'];
export type Game = Database['public']['Tables']['games']['Row'];
export type MatchmakingEntry = Database['public']['Tables']['matchmaking_queue']['Row'];
