# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the **big2** repository - a Deep Monte Carlo AI for the Big 2 card game.

**Current Stage**: Stage 1+ - Enhanced Pipeline with Parallel Training
**Goal**: Create a working training pipeline that beats random play (>50% win rate)
**Status**: Implementation complete with 6-worker parallelization

## Architecture

This is a simplified Deep Monte Carlo implementation optimized for M3 MacBook Air development with cloud GPU scaling capability.

### Key Components

1. **Game Environment** (`big2_ai/env/`)
   - `game.py`: Big 2 game rules and state management
   - `move_generator.py`: Legal move generation
   - `move_detector.py`: Move type classification
   - `encoding.py`: State/action encoding

2. **Neural Network** (`big2_ai/models/`)
   - `simple_network.py`: Dense network (no LSTM in Stage 1)
   - Architecture: 3x256 hidden layers
   - Input: state (143 dims) + action (52 dims)
   - Output: Q-value

3. **Training** (`big2_ai/training/`)
   - `buffer.py`: Replay buffer for Monte Carlo transitions
   - `trainer.py`: Parallel self-play training loop (6 workers default)

4. **Entry Points**
   - `train.py`: Training script with worker parallelization
   - `evaluate.py`: Evaluation script
   - `play.py`: Interactive play against trained AI
   - `config.py`: Hyperparameters

### State Encoding (143 dims)

```
- hand:             52 dims (one-hot: cards in hand)
- last_move:        52 dims (one-hot: cards in last non-pass move)
- opponent_counts:  39 dims (3 opponents × 13 card counts)
```

### Action Encoding (52 dims)

```
- action: 52 dims (one-hot: cards in the action)
```

### Key Design Decisions

- **Parallel self-play**: 6 workers (configurable) for faster data collection
- **Dense network only**: No LSTM, simpler architecture
- **Simplified state**: No full history, just last move
- **Small buffer**: 10K transitions, fits in RAM
- **High exploration**: Start at ε=0.9, decay slowly
- **CPU workers**: Workers use CPU for multiprocessing (MPS in main process)

### Usage

```bash
# Install dependencies
pip install -r requirements.txt

# View configuration
python train.py --config

# Train model with 6 workers (50K episodes, ~1-2 hours on M3 Air)
python train.py --episodes 50000

# Train with custom number of workers
python train.py --episodes 50000 --workers 8

# Train single-threaded (for debugging)
python train.py --episodes 50000 --workers 0

# Resume training
python train.py --resume --checkpoint checkpoints/big2_model.pt

# Evaluate model
python evaluate.py checkpoints/big2_model_best.pt --games 100

# Play against trained AI
python play.py checkpoints/big2_model_best.pt
```

### Testing Components

Each module has a built-in test:

```bash
# Test game engine
python -m big2_ai.env.game

# Test move detector
python -m big2_ai.env.move_detector

# Test move generator
python -m big2_ai.env.move_generator

# Test encoding
python -m big2_ai.env.encoding

# Test network
python -m big2_ai.models.simple_network

# Test buffer
python -m big2_ai.training.buffer

# Test trainer components
python -m big2_ai.training.trainer
```

### Critical Implementation Details

1. **All-player reward tracking**: The game returns all 4 players' rewards in the info dict, ensuring correct credit assignment.

2. **Trajectory storage**: Each player's (state, action) pairs are stored separately and backfilled with their episode return.

3. **Epsilon-greedy exploration**: High initial epsilon (0.9) with slow decay (0.9999 per episode).

4. **Target network**: Soft updates every 100 episodes (τ=0.01) for stability.

### Expected Performance

| Milestone | Win Rate vs Random |
|-----------|-------------------|
| Episode 1K | 40-50% |
| Episode 5K | 60-70% |
| Episode 20K | 70-80% |
| Episode 50K | 75-85% |

### Stage 2 Plans (Future)

After achieving >70% win rate:
- Add LSTM for history (16 previous moves)
- ✅ Parallelize with 6 workers (implemented, configurable)
- Better state encoding (add opponent_union)
- More sophisticated exploration

### Stage 3 Plans (Advanced)

After Stage 2:
- PTIE with critic network
- Hyperparameter optimization
- Opponent diversity training

## File Structure

```
big2/
├── big2_ai/
│   ├── __init__.py
│   ├── config.py              # Training hyperparameters
│   │
│   ├── env/
│   │   ├── __init__.py
│   │   ├── game.py            # Big 2 game rules
│   │   ├── move_generator.py # Legal move enumeration
│   │   ├── move_detector.py  # Move type classification
│   │   └── encoding.py        # State/action encoding
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   └── simple_network.py # Dense network (no LSTM)
│   │
│   └── training/
│       ├── __init__.py
│       ├── buffer.py          # Replay buffer
│       └── trainer.py         # Parallel training loop (6 workers)
│
├── train.py                   # Training entry point (with --workers)
├── evaluate.py                # Evaluation script
├── play.py                    # Interactive play against AI
├── requirements.txt
└── CLAUDE.md
```

## Development Guidelines

1. **Keep it simple**: Resist urge to add features
2. **Test incrementally**: Test each component in isolation
3. **Document assumptions**: Use assertions and comments
4. **Profile before optimizing**: Don't optimize prematurely
5. **Verify correctness**: Ensure all 4 players get correct rewards

## License

This project is licensed under the MIT License (see LICENSE file).
