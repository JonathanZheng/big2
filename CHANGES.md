# Changes

---

## Web App Feature Update - Online Multiplayer, Test Mode & Stats Tracking

### 1. Profile Stats Tracking Fix

**Problem**: Games completed against bots didn't update player statistics (games_played, games_won, ELO rating).

**Root Cause**: `GameBoard.tsx` had an `onGameEnd` callback prop, but `/src/app/play/bot/page.tsx` never passed a handler.

**Files Created**:
- `src/app/play/bot/actions.ts` - Server action `recordGameResult()` to update player stats

**Files Modified**:
- `src/app/play/bot/page.tsx` - Added `onGameEnd` handler that calls the server action
- `src/components/game/GameBoard.tsx` - Added `onNewGame` callback prop

**Features**:
- ELO calculation with K-factor of 32
- Win streak and best win streak tracking
- Highest ELO tracking
- ELO change notification after each game

### 2. Test Mode (AI Move Suggestion)

Allows users to input their hand and game history to get AI-suggested best moves for physical card games.

**Files Created**:
- `src/components/test-mode/CardSelector.tsx` - 52-card selection grid (13 ranks × 4 suits)
- `src/components/test-mode/HistoryInput.tsx` - Game history tracking with move validation
- `src/components/test-mode/SuggestionDisplay.tsx` - AI recommendation display with reasoning
- `src/components/test-mode/index.ts` - Component exports
- `src/lib/game/test-mode.ts` - Core analysis logic using greedy bot

**Files Modified**:
- `src/app/play/test/page.tsx` - Complete rewrite with analysis UI
- `src/lib/game/index.ts` - Added test-mode exports

**Features**:
- Select up to 13 cards for your hand
- Track full game history (all cards played by all players)
- AI suggests best move with reasoning
- Shows alternative moves ranked by score
- Position selector (0-3)

### 3. Online Multiplayer

Real-time 4-player games with room codes and ELO-based matchmaking.

**Database Migration** (`supabase/migrations/002_online_multiplayer.sql`):
- Added columns: `player_connections`, `turn_started_at`, `turn_timeout_seconds`, `is_private`
- Created functions: `join_game_room()`, `start_game()`, `find_match()`, `leave_matchmaking()`
- Updated RLS policies for multiplayer access

**Files Created**:
- `src/app/play/online/actions.ts` - Server actions:
  - `createRoom()` - Generate 6-character room code
  - `joinRoom()` - Join by room code
  - `joinMatchmaking()` - ELO-based queue matching (±200 range)
  - `leaveMatchmaking()` - Exit queue
  - `startGame()` - Host starts when 4 players joined
  - `makeMove()` - Server-side move validation
  - `leaveRoom()` - Leave waiting room
- `src/hooks/useMultiplayerGame.ts` - Real-time game state subscription
- `src/components/multiplayer/Lobby.tsx` - Create/join room and matchmaking UI
- `src/components/multiplayer/WaitingRoom.tsx` - Pre-game lobby with player slots
- `src/components/multiplayer/MultiplayerGameBoard.tsx` - Online game board with turn timer
- `src/components/multiplayer/index.ts` - Component exports

**Files Modified**:
- `src/app/play/online/page.tsx` - Complete rewrite with phases: lobby → waiting → playing → ended

**Features**:
- Private rooms with shareable 6-character codes
- ELO-based matchmaking (matches players within ±200 ELO)
- 30-second turn timer
- Real-time game synchronization via Supabase Realtime
- Server-side move validation to prevent cheating
- Stats update for all players after game completion

---

## Rule-Based Bot Integration & Evaluation Enhancements

### Code Cleanup

**File**: `big2_ai/agents/rule_based_bot.py`

- Removed unused imports: `detect_move_type`, `MoveType`

### Bot-vs-Bot Evaluation

**File**: `evaluate.py`

- **ADDED** `run_bot_vs_bot_evaluation()` function
- **ADDED** `--bot-vs-bot` CLI argument accepting 2 bot types

**Usage**:
```bash
python3 evaluate.py --bot-vs-bot rule_based_bot greedy_bot --games 200
```

### Training Opponent Distribution Update

**Files**: `big2_ai/training/trainer.py`, `big2_ai/config.py`

- **ADDED** `worker_play_episode_vs_rule_based()` worker function
- **CHANGED** default opponent distribution:
  - Self-play: 60%
  - Greedy opponents: 20%
  - Rule-based opponents: 20%
  - Random opponents: 0% (REMOVED)

---

## Training Dynamics & Move Generator Optimization

### Training Dynamics Fixes

**File**: `big2_ai/config.py`

- `epsilon_end`: 0.01 → 0.05
- `epsilon_decay_cutoff`: 0.9 → 0.95
- `learning_rate`: 5e-5 → 1e-4
- `batch_size`: 512 → 256

**File**: `big2_ai/training/trainer.py`

- Fixed PTIE bug: actor now uses baseline-subtracted returns

### Move Generator Optimization

**File**: `big2_ai/env/move_generator.py`

- Complete rewrite with rank/suit map optimization
- Expected speedup: 3-10x for move generation

---

## Evaluation & Checkpoint Improvements

### Changes Made

- `eval_freq`: 100 → 1000
- `eval_games`: 100 → 200
- Added top-K checkpoint management (keeps 5 best by greedy win rate)
- Fixed training ratios: 60% self-play, 20% greedy, 20% league

---

## State Encoding Overhaul

### Critical Bug Fix: Remove Hidden Information Leakage

**Problem**: Model was "cheating" by reading opponent hands via `encode_opponent_counts()`.

### Changes Made

- **REMOVED** `encode_opponent_counts()` (39 dims of cheating)
- **ADDED** `encode_graveyard()` (52 dims - cards played to discard)
- **ADDED** `encode_control_state()` (5 dims - trick control + passes)
- **FIXED** `encode_high_cards_in_graveyard()` (counts in graveyard only)

**State Dimensions**: 149 → 167 (legitimate information only)

---

## League Training Implementation

- `LeagueOpponent` dataclass and `League` class for opponent pool management
- Skill-matched sampling based on win rate similarity
- Periodic snapshots added to pool

---

## PTIE Implementation

Perfect Training for Imperfect-Information Games (from DouZero):

- `encode_perfect_state()` (321 dims - all hands visible)
- Critic network for value estimation with perfect information
- Advantage-based actor updates

---

## Web App Foundation

### Game Engine (TypeScript Port)
- `lib/game/types.ts` - Type definitions
- `lib/game/constants.ts` - Card utilities
- `lib/game/game-engine.ts` - Core Big2Game class
- `lib/game/move-detector.ts` - Move type detection
- `lib/game/move-generator.ts` - Legal move generation
- `lib/game/greedy-bot.ts` - AI bot with heuristics

### UI Components
- `components/game/Card.tsx` - Card display
- `components/game/PlayerHand.tsx` - Hand layouts
- `components/game/PlayArea.tsx` - Center play area
- `components/game/MoveControls.tsx` - Action buttons
- `components/game/GameBoard.tsx` - Main game container

### Auth & Database (Supabase)
- Login/Signup pages with username
- Profile page with stats display
- Database schema with RLS policies

---

## Bug Fixes

1. **Bot Not Making First Move** - Fixed React state detection with `gameVersion` counter
2. **Greedy Bot Typo** - Fixed `return 0;0;` → `return 0;`
3. **TypeScript Type Errors** - Added proper type casts for hand analysis

---

## Evaluation Tools

- `--pass_diagnosis` flag for passing behavior analysis
- `--models` flag for multi-model evaluation (up to 4 models)
