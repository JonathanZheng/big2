# Changes

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

## File Structure

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
