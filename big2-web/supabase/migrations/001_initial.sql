-- Big 2 Web Application - Initial Database Schema

-- User profiles (extends Supabase auth)
CREATE TABLE profiles (
  id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
  username TEXT UNIQUE NOT NULL,
  display_name TEXT,
  avatar_url TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Player statistics
CREATE TABLE player_stats (
  user_id UUID PRIMARY KEY REFERENCES profiles(id) ON DELETE CASCADE,
  games_played INTEGER DEFAULT 0,
  games_won INTEGER DEFAULT 0,
  elo_rating INTEGER DEFAULT 1000,
  highest_elo INTEGER DEFAULT 1000,
  win_streak INTEGER DEFAULT 0,
  best_win_streak INTEGER DEFAULT 0,
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Games
CREATE TABLE games (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  mode TEXT NOT NULL CHECK (mode IN ('bot', 'multiplayer', 'test')),
  status TEXT DEFAULT 'waiting' CHECK (status IN ('waiting', 'in_progress', 'completed', 'abandoned')),
  room_code TEXT UNIQUE,
  player_0_id UUID REFERENCES profiles(id),
  player_1_id UUID REFERENCES profiles(id),
  player_2_id UUID REFERENCES profiles(id),
  player_3_id UUID REFERENCES profiles(id),
  bot_difficulties JSONB,
  state JSONB NOT NULL,
  winner_position INTEGER,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Matchmaking queue
CREATE TABLE matchmaking_queue (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
  elo_rating INTEGER NOT NULL,
  joined_at TIMESTAMPTZ DEFAULT NOW(),
  status TEXT DEFAULT 'waiting' CHECK (status IN ('waiting', 'matched', 'cancelled'))
);

-- Create indexes for performance
CREATE INDEX idx_games_room_code ON games(room_code) WHERE room_code IS NOT NULL;
CREATE INDEX idx_games_status ON games(status);
CREATE INDEX idx_matchmaking_status ON matchmaking_queue(status) WHERE status = 'waiting';
CREATE INDEX idx_matchmaking_elo ON matchmaking_queue(elo_rating) WHERE status = 'waiting';

-- Enable Row Level Security
ALTER TABLE profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE player_stats ENABLE ROW LEVEL SECURITY;
ALTER TABLE games ENABLE ROW LEVEL SECURITY;
ALTER TABLE matchmaking_queue ENABLE ROW LEVEL SECURITY;

-- Profiles policies
CREATE POLICY "Public profiles are viewable by everyone"
  ON profiles FOR SELECT
  USING (true);

CREATE POLICY "Users can insert their own profile"
  ON profiles FOR INSERT
  WITH CHECK (auth.uid() = id);

CREATE POLICY "Users can update their own profile"
  ON profiles FOR UPDATE
  USING (auth.uid() = id);

-- Player stats policies
CREATE POLICY "Player stats are viewable by everyone"
  ON player_stats FOR SELECT
  USING (true);

CREATE POLICY "System can insert player stats"
  ON player_stats FOR INSERT
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "System can update player stats"
  ON player_stats FOR UPDATE
  USING (auth.uid() = user_id);

-- Games policies
CREATE POLICY "Games are viewable by participants"
  ON games FOR SELECT
  USING (
    auth.uid() IN (player_0_id, player_1_id, player_2_id, player_3_id)
    OR status = 'waiting'
  );

CREATE POLICY "Authenticated users can create games"
  ON games FOR INSERT
  WITH CHECK (auth.uid() IS NOT NULL);

CREATE POLICY "Game participants can update games"
  ON games FOR UPDATE
  USING (auth.uid() IN (player_0_id, player_1_id, player_2_id, player_3_id));

-- Matchmaking queue policies
CREATE POLICY "Users can view their own queue entries"
  ON matchmaking_queue FOR SELECT
  USING (auth.uid() = user_id);

CREATE POLICY "Users can insert themselves into queue"
  ON matchmaking_queue FOR INSERT
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own queue entries"
  ON matchmaking_queue FOR UPDATE
  USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own queue entries"
  ON matchmaking_queue FOR DELETE
  USING (auth.uid() = user_id);

-- Function to automatically create profile and stats on user signup
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER AS $$
BEGIN
  INSERT INTO public.profiles (id, username, display_name)
  VALUES (
    NEW.id,
    COALESCE(NEW.raw_user_meta_data->>'username', 'player_' || substr(NEW.id::text, 1, 8)),
    COALESCE(NEW.raw_user_meta_data->>'display_name', NEW.raw_user_meta_data->>'username', 'Player')
  );

  INSERT INTO public.player_stats (user_id)
  VALUES (NEW.id);

  RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Trigger to call function on new user signup
CREATE OR REPLACE TRIGGER on_auth_user_created
  AFTER INSERT ON auth.users
  FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();

-- Function to update timestamps
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Triggers for updated_at
CREATE TRIGGER update_profiles_updated_at
  BEFORE UPDATE ON profiles
  FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER update_player_stats_updated_at
  BEFORE UPDATE ON player_stats
  FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER update_games_updated_at
  BEFORE UPDATE ON games
  FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- Enable realtime for games table
ALTER PUBLICATION supabase_realtime ADD TABLE games;
