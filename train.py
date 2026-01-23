"""Entry point for training Big 2 AI."""

import argparse
from pathlib import Path

from big2_ai.training import train
from big2_ai.config import print_config, TRAINING_CONFIG


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Big 2 AI")

    parser.add_argument(
        "--episodes",
        type=int,
        default=100000,
        help="Number of training episodes (default: 100000)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/big2_model.pt",
        help="Path to save/load checkpoint (default: checkpoints/big2_model.pt)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from checkpoint"
    )
    parser.add_argument(
        "--config",
        action="store_true",
        help="Print configuration and exit"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        help="Number of parallel workers (default: 6, set to 0 for single-threaded)"
    )
    parser.add_argument(
        "--checkpoint-opponent",
        type=str,
        default=None,
        help="Path to frozen opponent model for curriculum learning (must use new 167-dim encoding)"
    )
    parser.add_argument(
        "--use-league",
        action="store_true",
        help="Enable league training with opponent pool (skill-matched sampling)"
    )
    parser.add_argument(
        "--league-dir",
        type=str,
        default="checkpoints/league",
        help="Directory for league checkpoints (default: checkpoints/league)"
    )
    parser.add_argument(
        "--snapshot-freq",
        type=int,
        default=None,
        help="Episodes between league snapshots (default: 2000)"
    )

    args = parser.parse_args()

    # Print config if requested
    if args.config:
        print_config()
        return

    # Override snapshot frequency in config if provided
    if args.snapshot_freq is not None:
        TRAINING_CONFIG["league_snapshot_freq"] = args.snapshot_freq

    # Print training info
    print("\n" + "=" * 60)
    print("Big 2 AI Training")
    print("=" * 60)
    print(f"Episodes: {args.episodes}")
    print(f"Workers: {args.workers}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Resume: {args.resume}")
    if args.checkpoint_opponent:
        print(f"Checkpoint opponent: {args.checkpoint_opponent}")
    if args.use_league:
        print(f"League training: ENABLED")
        print(f"League directory: {args.league_dir}")
    print("=" * 60 + "\n")

    # Train
    model, best_win_rate = train(
        num_episodes=args.episodes,
        checkpoint_path=args.checkpoint,
        resume=args.resume,
        num_workers=args.workers,
        checkpoint_opponent_path=args.checkpoint_opponent,
        use_league=args.use_league,
        league_dir=args.league_dir
    )

    print(f"\nTraining complete! Best win rate: {best_win_rate*100:.1f}%")
    print(f"Model saved to: {args.checkpoint}")


if __name__ == "__main__":
    main()
