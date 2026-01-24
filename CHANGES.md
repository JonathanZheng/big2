# Changes

---

## Training Dynamics & Move Generator Optimization (2025-01-24)

### Training Dynamics Fixes

**File**: `big2_ai/config.py`

- `epsilon_end`: 0.01 → 0.05 (maintain 5% exploration)
- `epsilon_decay_cutoff`: 0.9 → 0.95 (extend decay period)
- `learning_rate`: 5e-5 → 1e-4 (2x faster convergence)
- `batch_size`: 512 → 256 (more frequent updates)

**File**: `big2_ai/training/trainer.py`

- Fixed PTIE bug: actor now uses baseline-subtracted returns (`returns - critic_baseline`)
- Removed unused advantage computation (was computed but never used)

### Move Generator Optimization

**File**: `big2_ai/env/move_generator.py`

Complete rewrite with rank/suit map optimization:

- **Build maps once**: `build_maps(hand)` creates rank_map and suit_map at start
- **Direct combo generation**:
  - Pairs/Triples: O(1) lookup via rank_map instead of scanning hand
  - Straights: Only check 9 possible consecutive rank sequences (not C(13,5))
  - Flushes: Only check within same-suit groups (max C(7,5) per suit)
  - Full houses: Direct 3+2 pattern matching via rank_map
  - Quads: O(1) lookup for ranks with exactly 4 cards
- **Conditional generation**: Only generate 5-card combos when `last_move` is 5-card or free turn

**Expected speedup**: 3-10x for move generation (main training bottleneck)

---

## Evaluation & Checkpoint Improvements (2025-01-23)

### Changes Made

#### 1. Evaluation Streamlined

**File**: `big2_ai/config.py`

- `eval_freq`: 100 → 1000 (evaluate less often for faster training)
- `eval_games`: 100 → 200 (95% CI ~±6.4% at 30% WR)

**File**: `big2_ai/training/trainer.py`

- Removed evaluation against random bots (only evaluate vs greedy bots)
- Significantly faster training loop

#### 2. Top-K Checkpoint Saving

**File**: `big2_ai/training/trainer.py`

- Added top-K checkpoint management to keep the 5 best checkpoints ranked by greedy win rate
- Tie-breaker: newer models (higher episode) preferred when win rates are equal
- Automatically removes old checkpoints when pool exceeds K
- Each checkpoint is named `big2_model_top_{episode}.pt`

**File**: `big2_ai/config.py`

- Added `top_k_checkpoints: 5` configuration option

#### 3. Fixed Training Ratios for League

**File**: `big2_ai/config.py`

- Fixed ratio: 60% self-play, 20% vs greedy, 20% vs league opponents
- Removed progressive curriculum phases (was shifting ratios over time)

### Directory Structure

```
checkpoints/
├── big2_model.pt           # Latest checkpoint (periodic)
├── big2_model_best.pt      # Single best checkpoint
├── big2_model_top_1000.pt  # Top checkpoint from episode 1000
├── big2_model_top_2000.pt  # Top checkpoint from episode 2000
├── ...                     # (up to 5 top checkpoints)
└── league/                 # League training checkpoints (updated every 2000 episodes)
```

---

## State Encoding Overhaul (2025-01-23)

### Critical Bug Fix: Remove Hidden Information Leakage

**Problem Discovered**: The model was "cheating" during training by reading opponent hands directly via `encode_opponent_counts()`. This function accessed `game.hands[i]` for opponents, giving the model perfect information about what cards opponents held — information that would never be available in real play.

**Impact**:
- Training/test distribution mismatch
- Model learned to rely on hidden information
- Performance gains were artificially inflated
- Model would fail against real opponents or if cheating was removed

### Changes Made

#### 1. Removed Cheating Observations

**File**: `big2_ai/env/encoding.py`

- **REMOVED** `encode_opponent_counts()` from state encoding
  - This function directly read `game.hands[i]` for opponents (cheating)
  - Was contributing 39 dimensions of illegitimate information

#### 2. Added Graveyard Encoding (NEW)

**File**: `big2_ai/env/encoding.py`

- **ADDED** `encode_graveyard()` function
  - 52-dim one-hot vector of cards played to discard pile
  - Computed from `game.move_history` (legitimate observable information)
  - Critical for strategic play: knowing which cards are "dead"

#### 3. Added Turn/Control State (NEW)

**File**: `big2_ai/env/encoding.py`

- **ADDED** `encode_control_state()` function
  - `control_player`: 4-dim one-hot (who controls the trick, relative to self)
  - `consecutive_passes`: 1-dim normalized (passes since last play / 3.0)
  - Essential for knowing when you can play freely vs must beat last move

#### 4. Fixed High Cards Encoding

**File**: `big2_ai/env/encoding.py`

- **MODIFIED** `encode_remaining_high_cards()` → `encode_high_cards_in_graveyard()`
  - Now counts high cards (2s, Aces, Kings) **in the graveyard only**
  - Previously counted "cards not in hand" which was ambiguous
  - Model can now compute: `unknown = 4 - in_hand - in_graveyard`

#### 5. Updated State Dimensions

**Old State (149 dims)** - WITH CHEATING:
| Feature | Dims | Status |
|---------|------|--------|
| Hand | 52 | Kept |
| Last Move | 52 | Kept |
| Opponent Counts | 39 | **REMOVED (cheating)** |
| Opponent Hand Sizes | 3 | Kept |
| Remaining High Cards | 3 | **FIXED** |

**New State (167 dims)** - LEGITIMATE:
| Feature | Dims | Status |
|---------|------|--------|
| Hand | 52 | Unchanged |
| Graveyard | 52 | **NEW** |
| Last Move | 52 | Unchanged |
| Control Player | 4 | **NEW** |
| Consecutive Passes | 1 | **NEW** |
| Opponent Hand Sizes | 3 | Unchanged |
| High Cards in Graveyard | 3 | **FIXED** |

#### 6. Updated Configuration

- `big2_ai/config.py`: `state_dim`: 149 → 167
- `big2_ai/models/simple_network.py`: default state_dim updated
- `big2_ai/models/lstm_network.py`: default state_dim updated

### Migration Notes

- **Existing checkpoints are incompatible** with the new state encoding
- Retrain from scratch with the corrected encoding
- Initial performance may appear lower but represents legitimate capability

---

## League Training Implementation (2025-01-23)

### Opponent Pool with Skill-Matched Sampling

League training maintains a pool of historical checkpoints as opponents, providing diverse training experience and preventing overfitting to a single opponent.

### Changes Made

#### 1. League Module

**File**: `big2_ai/training/league.py` (NEW)

- `LeagueOpponent` dataclass: stores generation, filepath, win_rate
- `League` class: manages opponent pool
  - `add_snapshot()`: Add current model to pool
  - `sample_opponent()`: Skill-matched sampling (prefers opponents with similar win rates)
  - `load_from_disk()` / `save_metadata()`: Persistence via metadata.json
  - Automatic pruning when pool exceeds max_opponents

#### 2. Trainer Integration

**File**: `big2_ai/training/trainer.py`

- Added `use_league` and `league_dir` parameters to `train()`
- League initialization with optional initial snapshot
- Workers sample different opponents from pool each iteration
- Periodic snapshots added to pool based on `league_snapshot_freq`
- Win rate tracking for skill-matched sampling

#### 3. CLI Arguments

**File**: `train.py`

```bash
--use-league          # Enable league training
--league-dir DIR      # Directory for league checkpoints (default: checkpoints/league)
--snapshot-freq N     # Episodes between snapshots (default: 2000)
```

#### 4. Configuration

**File**: `big2_ai/config.py`

```python
"use_league": False,           # Enable league training
"league_dir": "checkpoints/league",
"league_max_opponents": 20,    # Max opponents in pool
"league_snapshot_freq": 2000,  # Episodes between snapshots
"league_initial_snapshot": True,
```

### Directory Structure

```
checkpoints/league/
├── metadata.json      # Opponent metadata (generation, win_rate, filepath)
├── gen_000000.pt      # Initial snapshot
├── gen_002000.pt      # Snapshot at episode 2000
├── gen_004000.pt      # etc.
```

### Skill-Matched Sampling

Opponents are sampled with probability inversely proportional to their win rate distance from the current agent:

```python
weight = 1.0 / (|opponent_win_rate - current_win_rate| + 0.1)
```

This ensures the agent trains against appropriately challenging opponents.

---

## PTIE Implementation (2025-01-23)

### Perfect Training for Imperfect-Information Games (PTIE)

PTIE is a training technique from DouZero that uses a critic network with perfect information (all hands visible) to guide the actor network which only sees partial information.

### Changes Made

#### 1. Perfect State Encoding

**File**: `big2_ai/env/encoding.py`

- **ADDED** `encode_perfect_state()` function (321 dims)
  - All 4 hands encoded (208 dims: 4 × 52)
  - Graveyard (52 dims)
  - Last move (52 dims)
  - Control player (4 dims one-hot)
  - Consecutive passes (1 dim)
  - Current player (4 dims one-hot)
- **ADDED** `PERFECT_STATE_DIM = 321` constant

#### 2. Critic Network

**File**: `big2_ai/models/critic_network.py` (NEW)

- 3-layer MLP for value estimation
- Input: 321-dim perfect state
- Hidden: 256 units per layer with ReLU activation
- Output: scalar value estimate

#### 3. Training Buffer Updates

**File**: `big2_ai/training/buffer.py`

- Added `perfect_state` field to Transition dataclass
- Updated `push()` to accept optional `perfect_state`
- Updated `sample_arrays()` to return perfect_states when `include_perfect=True`

#### 4. Trainer Updates

**File**: `big2_ai/training/trainer.py`

- Added critic network creation alongside actor
- Updated all worker functions to collect perfect states:
  - `worker_play_episode()` (self-play)
  - `worker_play_episode_vs_greedy()` (greedy opponents)
  - `worker_play_episode_vs_random()` (random opponents)
  - `worker_play_episode_vs_checkpoint()` (checkpoint opponents)
- Added critic training loop with MSE loss
- Updated checkpointing to save/load critic state
- Updated logging to show both ActorL and CriticL

#### 5. Configuration

**File**: `big2_ai/config.py`

```python
TRAINING_CONFIG = {
    ...
    "use_ptie": True,
    "critic_learning_rate": 1e-4,
    "critic_weight": 0.5,
    "advantage_weight": 1.0,
}

CRITIC_CONFIG = {
    "perfect_state_dim": 321,
    "hidden_dim": 256,
}
```

### How PTIE Works

1. During episode collection, both partial state (167 dims) and perfect state (321 dims) are recorded
2. Critic network estimates value V(s_perfect) using perfect information
3. Advantage A = R - V(s_perfect) guides actor updates
4. Actor learns from returns but benefits from lower-variance critic baselines

---

## Phase 1 & 2: Foundation + Play vs Bots

### Game Engine (TypeScript Port from Python)
- `lib/game/types.ts` - Type definitions for cards, moves, game state
- `lib/game/constants.ts` - Card utilities (rank, suit, string conversion)
- `lib/game/game-engine.ts` - Core Big2Game class (from `game.py`)
- `lib/game/move-detector.ts` - Move type detection and comparison (from `move_detector.py`)
- `lib/game/move-generator.ts` - Legal move generation (from `move_generator.py`)
- `lib/game/greedy-bot.ts` - AI bot with strategic heuristics (from `greedy_bot.py`)

### UI Components
- `components/game/Card.tsx` - Individual card display with suit colors
- `components/game/PlayerHand.tsx` - Fan display of cards with position-based layouts
- `components/game/PlayArea.tsx` - Center area showing last played move
- `components/game/MoveControls.tsx` - Play/Pass/Clear buttons with validation
- `components/game/GameBoard.tsx` - Main game container with bot turn automation

### Pages
- `app/page.tsx` - Landing page with mode selection
- `app/play/bot/page.tsx` - Play vs Bots game page
- `app/play/online/page.tsx` - Online multiplayer placeholder
- `app/play/test/page.tsx` - Test mode placeholder

---

## Bug Fixes

### 1. Bot Not Making First Move (`big2-web/src/components/game/GameBoard.tsx`)
- **Root cause:** React wasn't detecting state changes because `setGame(game)` passed the same object reference. Additionally, the timer was being cleared when the effect re-ran due to state changes.
- **Fix:**
  - Added `gameVersion` counter state to force re-renders when game state changes
  - Added `botTimerRef` ref to prevent timer from being cleared when `isProcessingBotTurn` state changed
  - Removed `isProcessingBotTurn` from useEffect dependency array

### 2. Greedy Bot Typo (`big2-web/src/lib/game/greedy-bot.ts`)
- Fixed `return 0;0;` → `return 0;` (line 136)

### 3. TypeScript Type Errors (`big2-web/src/lib/game/greedy-bot.ts`)
- Added `as Rank[]` type casts for hand analysis arrays
- Added `as Rank` casts in `breaksCombination` function

---

## UI Improvements

### 4. Card Overlap (`big2-web/src/components/game/PlayerHand.tsx`)
- Reduced human player card overlap from `-ml-8` to `-ml-3`
- Cards now show rank/suit without being blocked by adjacent cards

---

## Phase 3: Auth & Database (Supabase Integration)

### New Files Created:
- `.env.local` - Supabase credentials (NEXT_PUBLIC_SUPABASE_URL, NEXT_PUBLIC_SUPABASE_ANON_KEY)
- `src/lib/supabase/client.ts` - Browser client
- `src/lib/supabase/server.ts` - Server client for Server Components
- `src/lib/supabase/middleware.ts` - Session refresh helper
- `src/lib/supabase/types.ts` - TypeScript types for database schema
- `src/middleware.ts` - Next.js middleware for auth session handling
- `supabase/migrations/001_initial.sql` - Database schema with RLS policies
- `src/app/auth/login/page.tsx` - Login page
- `src/app/auth/signup/page.tsx` - Signup page with username
- `src/app/auth/callback/route.ts` - OAuth callback handler
- `src/app/profile/page.tsx` - Profile page with stats display
- `src/app/profile/logout-button.tsx` - Logout button component

### Modified Files:
- `src/app/page.tsx` - Added auth-aware header (Login/Signup or Profile button)

### Dependencies Added:
- `@supabase/supabase-js`
- `@supabase/ssr`

---

## Evaluation Tools

### 5. Pass Diagnosis Feature (`evaluate.py`)
- Added `--pass_diagnosis` flag to analyze bot passing behavior
- Tracks how often bots pass when they have other legal moves available
- Can run with or without a model checkpoint

### 6. Multi-Model Evaluation (`evaluate.py`)
- Added `--models` flag to evaluate multiple models against each other
- Accepts 1-4 model checkpoints; remaining slots filled with greedy bots
- Shows per-player results sorted by win rate
- Includes model rankings summary for head-to-head comparison

---

## File Structure for big2-web

```
big2-web/
├── src/
│   ├── app/
│   │   ├── page.tsx
│   │   ├── layout.tsx
│   │   ├── auth/
│   │   │   ├── login/page.tsx
│   │   │   ├── signup/page.tsx
│   │   │   └── callback/route.ts
│   │   ├── play/
│   │   │   ├── bot/page.tsx
│   │   │   ├── online/page.tsx
│   │   │   └── test/page.tsx
│   │   └── profile/
│   │       ├── page.tsx
│   │       └── logout-button.tsx
│   ├── components/
│   │   ├── game/
│   │   │   ├── Card.tsx
│   │   │   ├── PlayerHand.tsx
│   │   │   ├── PlayArea.tsx
│   │   │   ├── MoveControls.tsx
│   │   │   ├── GameBoard.tsx
│   │   │   └── index.ts
│   │   └── ui/
│   └── lib/
│       ├── game/
│       │   ├── types.ts
│       │   ├── constants.ts
│       │   ├── game-engine.ts
│       │   ├── move-detector.ts
│       │   ├── move-generator.ts
│       │   ├── greedy-bot.ts
│       │   └── index.ts
│       └── supabase/
│           ├── client.ts
│           ├── server.ts
│           ├── middleware.ts
│           └── types.ts
├── supabase/
│   └── migrations/
│       └── 001_initial.sql
└── package.json
```
