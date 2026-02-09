# Big 2 AI - Deep Monte Carlo Training

A Deep Monte Carlo implementation for learning to play Big 2 (a popular 4-player card game), with a full-featured web application.

## Quick Start

### Installation (Python AI Training)

```bash
# Install dependencies
pip install -r requirements.txt
```

### Training

```bash
# View configuration
python train.py --config

# Train with 6 parallel workers (default, recommended)
python train.py --episodes 50000

# Train with custom number of workers
python train.py --episodes 50000 --workers 8

# Train single-threaded (for debugging)
python train.py --episodes 50000 --workers 0

# Resume training from checkpoint
python train.py --resume --checkpoint checkpoints/big2_model.pt

# Train with curriculum learning (vs frozen checkpoint opponent)
python train.py --episodes 50000 --checkpoint_opponent checkpoints/baseline.pt
```

**Parallel Training:**
- Default: 6 workers play episodes in parallel
- Each batch generates 6x more diverse experience
- Workers use CPU (MPS doesn't support multiprocessing)
- Adjust `--workers` based on your CPU cores

### Evaluation

```bash
# Evaluate trained model
python evaluate.py checkpoints/big2_model_best.pt --games 100

# Verbose evaluation (shows first game)
python evaluate.py checkpoints/big2_model_best.pt --games 100 --verbose

# Pass diagnosis (analyze passing behavior)
python evaluate.py checkpoints/big2_model_best.pt --pass_diagnosis

# Multi-model evaluation (up to 4 models)
python evaluate.py --models model1.pt model2.pt model3.pt --games 100
```

### Interactive Play

```bash
# Play against trained AI (you are Player 0)
python play.py checkpoints/big2_model_best.pt

# Play as a different player position
python play.py checkpoints/big2_model_best.pt --player 2

# Play with reproducible seed
python play.py checkpoints/big2_model_best.pt --seed 42
```

**Game Commands:**
- Type card names to play them (e.g., `3d 4d 5d` for 3-4-5 of diamonds straight)
- Type `pass` or `p` to pass
- Type `help` to see all legal moves
- Type `quit` to exit the game

**Card Notation:**
- Ranks: `3`, `4`, `5`, `6`, `7`, `8`, `9`, `10`, `j`, `q`, `k`, `a`, `2`
- Suits: `d` (diamonds), `c` (clubs), `h` (hearts), `s` (spades)
- Examples: `3d`, `10h`, `ah`, `2s`
- Case-insensitive

## Web Application

The web app provides a browser-based interface to play Big 2 with multiple game modes.

### Running the Web App

```bash
cd big2-web

# Install dependencies
npm install

# Run development server
npm run dev
```

The app will be available at `http://localhost:3000`.

### Web App Features

- **Play vs Bots** - Single player against 3 AI opponents with ELO tracking
- **Online Multiplayer** - Real-time 4-player games with room codes or matchmaking
- **Test Mode** - AI move suggestions for analyzing physical card games
- **User Profiles** - Stats tracking (games played, wins, ELO rating, streaks)
- **Authentication** - Email/password login via Supabase

### Tech Stack

- **Frontend**: Next.js 16, React 19, Tailwind CSS
- **Backend**: Supabase (PostgreSQL, Auth, Realtime)
- **Game Engine**: TypeScript port of Python implementation

## Architecture Overview

### Current Features

This implementation includes:

- **Parallel self-play**: 6 workers (configurable) for faster data collection
- **4-layer Dense network**: 256 hidden units per layer (~198K parameters)
- **Legitimate state encoding**: 167 dimensions (no hidden information)
- **LSTM variant available**: For temporal reasoning over move history
- **Curriculum learning**: Mix of self-play, greedy bots, and checkpoint opponents
- **Monte Carlo learning**: Episode returns with margin-based rewards

### Components

```
big2/
├── play.py           # Interactive play against AI
├── train.py          # Training script
├── evaluate.py       # Evaluation script
├── CHANGES.md        # Change log
├── big2_ai/
│   ├── env/              # Game environment
│   │   ├── game.py       # Big 2 rules & state management
│   │   ├── move_generator.py   # Legal move generation
│   │   ├── move_detector.py    # Move type classification
│   │   └── encoding.py         # State/action encoding (167 dims)
│   ├── models/
│   │   ├── simple_network.py   # Dense Q-network (~198K params)
│   │   ├── lstm_network.py     # LSTM variant for history
│   │   └── critic_network.py   # Critic network for PTIE
│   ├── agents/
│   │   ├── greedy_bot.py       # Rule-based greedy opponent
│   │   └── rule_based_bot.py   # Research paper rule-based bot
│   └── training/
│       ├── buffer.py     # Replay buffer with normalization
│       ├── trainer.py    # Training loop with parallel workers
│       └── league.py     # League training opponent pool
└── big2-web/
    ├── src/
    │   ├── app/
    │   │   ├── page.tsx
    │   │   ├── layout.tsx
    │   │   ├── auth/           # Authentication pages
    │   │   ├── play/           # Game modes (bot, online, test)
    │   │   └── profile/        # User profile & stats
    │   ├── components/
    │   │   ├── game/           # Game UI components
    │   │   ├── multiplayer/    # Online multiplayer components
    │   │   ├── test-mode/      # Test mode components
    │   │   └── ui/             # Shared UI components
    │   ├── hooks/              # React hooks (useMultiplayerGame)
    │   └── lib/
    │       ├── game/           # Game engine (TypeScript port)
    │       └── supabase/       # Supabase integration
    ├── supabase/
    │   └── migrations/         # Database migrations
    └── package.json
```

### State Encoding (167 dims)

All information is **legitimately observable** by the player:

| Feature | Dims | Description |
|---------|------|-------------|
| Hand | 52 | One-hot encoding of cards in hand |
| Graveyard | 52 | One-hot of cards played to discard pile |
| Last Move | 52 | One-hot of last non-pass move |
| Control Player | 4 | One-hot of who controls trick (relative) |
| Consecutive Passes | 1 | Normalized count (passes / 3.0) |
| Opponent Hand Sizes | 3 | Normalized card counts per opponent |
| High Cards in Graveyard | 3 | Count of 2s/Aces/Kings in discard |

### Action Encoding (52 dims)

- One-hot encoding of cards in the move
- Pass is encoded as all zeros

## Testing

Each component has a built-in test:

```bash
# Test game engine
python -m big2_ai.env.game

# Test move detector
python -m big2_ai.env.move_detector

# Test move generator
python -m big2_ai.env.move_generator

# Test state encoding
python -m big2_ai.env.encoding

# Test neural network
python -m big2_ai.models.simple_network

# Test replay buffer
python -m big2_ai.training.buffer

# Test training components
python -m big2_ai.training.trainer
```

## Training Configuration

Key hyperparameters (see `big2_ai/config.py`):

| Parameter | Value | Description |
|-----------|-------|-------------|
| Episodes | 50,000 | Total training episodes |
| Buffer size | 100,000 | Replay buffer capacity |
| Batch size | 256 | Training batch size |
| Learning rate | 1e-4 | Adam optimizer LR |
| Epsilon | 0.9 → 0.05 | Cosine annealing exploration |
| Warmup | 1,000 | Episodes before epsilon decay |
| Target update | Every 100 eps | Soft update with tau=0.005 |
| Gradient clip | 5.0 | Max gradient norm |
| Margin rewards | On | Scale rewards with win margin |

### Curriculum Learning

Fixed opponent distribution for training:

| Opponent Type | Ratio |
|---------------|-------|
| Self-Play | 60% |
| Greedy Bot | 20% |
| Rule-Based Bot | 20% |

## Big 2 Rules

The game follows standard Big 2 rules:

- 52 cards, 4 players, 13 cards each
- First move must contain 3 of Diamonds
- Move types: Single, Pair, Triple, Straight (5), Flush (5), Full House, Quad+Kicker, Straight Flush
- Pass allowed after any non-pass move
- 3 consecutive passes = trick winner starts new round
- Game ends when any player runs out of cards
- Rank order: 3 < 4 < ... < K < A < 2
- Suit order: Diamonds < Clubs < Hearts < Spades

## License

MIT License (see LICENSE file)

---

**Note**: Previous checkpoints are incompatible due to state encoding changes. See `CHANGES.md` for migration details.
