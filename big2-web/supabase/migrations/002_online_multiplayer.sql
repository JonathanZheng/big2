-- Online Multiplayer Enhancement Migration

-- Add player connection tracking to games table
ALTER TABLE games ADD COLUMN IF NOT EXISTS player_connections JSONB DEFAULT '{}';

-- Add turn timeout tracking
ALTER TABLE games ADD COLUMN IF NOT EXISTS turn_started_at TIMESTAMPTZ;
ALTER TABLE games ADD COLUMN IF NOT EXISTS turn_timeout_seconds INTEGER DEFAULT 30;

-- Add game room visibility
ALTER TABLE games ADD COLUMN IF NOT EXISTS is_private BOOLEAN DEFAULT true;

-- Update matchmaking queue for better matching
ALTER TABLE matchmaking_queue ADD COLUMN IF NOT EXISTS elo_range INTEGER DEFAULT 200;

-- Create function for atomic player join
CREATE OR REPLACE FUNCTION join_game_room(
  p_game_id UUID,
  p_user_id UUID,
  p_position INTEGER
) RETURNS BOOLEAN AS $$
DECLARE
  v_column TEXT;
  v_current_player UUID;
BEGIN
  v_column := 'player_' || p_position || '_id';

  -- Lock the row and check if slot is available
  EXECUTE format('SELECT %I FROM games WHERE id = $1 FOR UPDATE', v_column)
  INTO v_current_player
  USING p_game_id;

  IF v_current_player IS NOT NULL THEN
    RETURN FALSE;
  END IF;

  -- Update the slot
  EXECUTE format('UPDATE games SET %I = $1 WHERE id = $2', v_column)
  USING p_user_id, p_game_id;

  RETURN TRUE;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Create function to start a game when all players are ready
CREATE OR REPLACE FUNCTION start_game(p_game_id UUID, p_initial_state JSONB)
RETURNS BOOLEAN AS $$
DECLARE
  v_game RECORD;
BEGIN
  SELECT * INTO v_game FROM games WHERE id = p_game_id FOR UPDATE;

  IF v_game IS NULL THEN
    RETURN FALSE;
  END IF;

  -- Check all 4 player slots are filled
  IF v_game.player_0_id IS NULL OR v_game.player_1_id IS NULL
     OR v_game.player_2_id IS NULL OR v_game.player_3_id IS NULL THEN
    RETURN FALSE;
  END IF;

  -- Check game is in waiting status
  IF v_game.status != 'waiting' THEN
    RETURN FALSE;
  END IF;

  -- Start the game
  UPDATE games
  SET status = 'in_progress',
      state = p_initial_state,
      turn_started_at = NOW()
  WHERE id = p_game_id;

  RETURN TRUE;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Create matchmaking function to find players within ELO range
CREATE OR REPLACE FUNCTION find_match(p_user_id UUID, p_elo INTEGER)
RETURNS UUID AS $$
DECLARE
  v_game_id UUID;
  v_matched_users UUID[];
  v_elo_range INTEGER := 200;
BEGIN
  -- Find 3 other players within ELO range who are waiting
  SELECT array_agg(user_id ORDER BY joined_at)
  INTO v_matched_users
  FROM (
    SELECT user_id
    FROM matchmaking_queue
    WHERE status = 'waiting'
      AND user_id != p_user_id
      AND ABS(elo_rating - p_elo) <= v_elo_range
    ORDER BY joined_at
    LIMIT 3
  ) sub;

  -- Need exactly 3 other players
  IF v_matched_users IS NULL OR array_length(v_matched_users, 1) != 3 THEN
    RETURN NULL;
  END IF;

  -- Create game with all 4 players
  INSERT INTO games (
    mode,
    status,
    state,
    player_0_id,
    player_1_id,
    player_2_id,
    player_3_id,
    is_private
  )
  VALUES (
    'multiplayer',
    'waiting',
    '{}',
    p_user_id,
    v_matched_users[1],
    v_matched_users[2],
    v_matched_users[3],
    false
  )
  RETURNING id INTO v_game_id;

  -- Update all matched players' queue status
  UPDATE matchmaking_queue
  SET status = 'matched'
  WHERE user_id = ANY(v_matched_users) OR user_id = p_user_id;

  RETURN v_game_id;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Create function to leave matchmaking queue
CREATE OR REPLACE FUNCTION leave_matchmaking(p_user_id UUID)
RETURNS BOOLEAN AS $$
BEGIN
  DELETE FROM matchmaking_queue
  WHERE user_id = p_user_id AND status = 'waiting';

  RETURN FOUND;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Index for faster matchmaking queries
CREATE INDEX IF NOT EXISTS idx_matchmaking_waiting_elo
ON matchmaking_queue(elo_rating, joined_at)
WHERE status = 'waiting';

-- Allow users to see matchmaking queue for matching
DROP POLICY IF EXISTS "Users can view waiting queue for matching" ON matchmaking_queue;
CREATE POLICY "Users can view waiting queue for matching"
  ON matchmaking_queue FOR SELECT
  USING (status = 'waiting' OR auth.uid() = user_id);

-- Update games policy to allow viewing waiting rooms
DROP POLICY IF EXISTS "Games are viewable by participants" ON games;
CREATE POLICY "Games are viewable by participants and waiting rooms"
  ON games FOR SELECT
  USING (
    auth.uid() IN (player_0_id, player_1_id, player_2_id, player_3_id)
    OR (status = 'waiting' AND is_private = false)
    OR (room_code IS NOT NULL AND status = 'waiting')
  );
