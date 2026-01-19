"""Entry point for training Big 2 AI."""

import argparse
from pathlib import Path

from big2_ai.training import train
from big2_ai.config import print_config


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Big 2 AI")

    parser.add_argument(
        "--episodes",
        type=int,
        default=50000,
        help="Number of training episodes (default: 50000)"
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

    args = parser.parse_args()

    # Print config if requested
    if args.config:
        print_config()
        return

    # Print training info
    print("\n" + "=" * 60)
    print("Big 2 AI Training")
    print("=" * 60)
    print(f"Episodes: {args.episodes}")
    print(f"Workers: {args.workers}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Resume: {args.resume}")
    print("=" * 60 + "\n")

    # Train
    model, best_win_rate = train(
        num_episodes=args.episodes,
        checkpoint_path=args.checkpoint,
        resume=args.resume,
        num_workers=args.workers
    )

    print(f"\nTraining complete! Best win rate: {best_win_rate*100:.1f}%")
    print(f"Model saved to: {args.checkpoint}")


if __name__ == "__main__":
    main()
