"""League training module for Big2 AI.

Manages a pool of historical opponent checkpoints for diverse training.
Implements skill-matched sampling to select appropriately challenging opponents.
"""

import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import torch


@dataclass
class LeagueOpponent:
    """Metadata for a league opponent."""
    generation: int           # Episode when snapshot was taken
    filepath: str             # Path to checkpoint file (relative to league_dir)
    win_rate: float           # Win rate when added to pool

    # Lazy-loaded weights (not serialized)
    _model_state_dict: Optional[Dict] = None

    def to_dict(self) -> Dict:
        """Convert to dict for JSON serialization (excludes weights)."""
        return {
            "generation": self.generation,
            "filepath": self.filepath,
            "win_rate": self.win_rate,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "LeagueOpponent":
        """Create from dict."""
        return cls(
            generation=data["generation"],
            filepath=data["filepath"],
            win_rate=data["win_rate"],
        )


class League:
    """Manages pool of historical opponent checkpoints for league training.

    Key features:
    - Maintains up to max_opponents snapshots
    - Skill-matched sampling: prefers opponents with similar win rates
    - Lazy loading of model weights to save memory
    - Persistent storage via metadata.json
    """

    def __init__(self, league_dir: str, max_opponents: int = 20):
        """Initialize league.

        Args:
            league_dir: Directory to store league checkpoints
            max_opponents: Maximum number of opponents to keep in pool
        """
        self.league_dir = Path(league_dir)
        self.max_opponents = max_opponents
        self.opponents: List[LeagueOpponent] = []
        self.metadata_path = self.league_dir / "metadata.json"

        # Ensure directory exists
        self.league_dir.mkdir(parents=True, exist_ok=True)

    def add_snapshot(
        self,
        model_state_dict: Dict,
        generation: int,
        win_rate: float
    ) -> str:
        """Add current training agent to the opponent pool.

        Args:
            model_state_dict: Model weights to save
            generation: Current episode number
            win_rate: Current win rate of the model

        Returns:
            Path to saved checkpoint
        """
        # Generate filename
        filename = f"gen_{generation:06d}.pt"
        filepath = self.league_dir / filename

        # Save checkpoint (just model weights, not full training state)
        torch.save({"model_state_dict": model_state_dict}, filepath)

        # Create opponent entry
        opponent = LeagueOpponent(
            generation=generation,
            filepath=filename,
            win_rate=win_rate,
        )
        opponent._model_state_dict = model_state_dict  # Cache weights

        self.opponents.append(opponent)

        # Prune if over capacity (remove oldest, keeping diverse skill levels)
        if len(self.opponents) > self.max_opponents:
            self._prune_opponents()

        # Save metadata
        self.save_metadata()

        return str(filepath)

    def _prune_opponents(self):
        """Remove excess opponents while maintaining skill diversity.

        Strategy: Keep the oldest (baseline), newest, and spread across skill levels.
        """
        if len(self.opponents) <= self.max_opponents:
            return

        # Always keep first (baseline) and last (most recent)
        keep_indices = {0, len(self.opponents) - 1}

        # Sort by win rate to get skill distribution
        sorted_by_skill = sorted(
            enumerate(self.opponents),
            key=lambda x: x[1].win_rate
        )

        # Keep evenly spaced by skill level
        n_to_keep = self.max_opponents - len(keep_indices)
        if n_to_keep > 0:
            step = len(sorted_by_skill) / n_to_keep
            for i in range(n_to_keep):
                idx = int(i * step)
                keep_indices.add(sorted_by_skill[idx][0])

        # Remove opponents not in keep set
        new_opponents = []
        for i, opp in enumerate(self.opponents):
            if i in keep_indices:
                new_opponents.append(opp)
            else:
                # Delete checkpoint file
                filepath = self.league_dir / opp.filepath
                if filepath.exists():
                    filepath.unlink()

        self.opponents = new_opponents

    def sample_opponent(self, current_win_rate: float = 0.5) -> Optional[LeagueOpponent]:
        """Sample an opponent using skill-matched strategy.

        Prefers opponents with win rates similar to the current agent's win rate.

        Args:
            current_win_rate: Current agent's win rate (0.0 to 1.0)

        Returns:
            Sampled opponent, or None if pool is empty
        """
        if not self.opponents:
            return None

        # Weight by inverse distance to current win rate
        weights = []
        for opp in self.opponents:
            distance = abs(opp.win_rate - current_win_rate)
            # +0.1 to avoid division by zero, ensures all opponents have some chance
            weight = 1.0 / (distance + 0.1)
            weights.append(weight)

        # Sample with computed weights
        opponent = random.choices(self.opponents, weights=weights, k=1)[0]

        # Ensure weights are loaded
        self._ensure_loaded(opponent)

        return opponent

    def _ensure_loaded(self, opponent: LeagueOpponent):
        """Ensure opponent's model weights are loaded."""
        if opponent._model_state_dict is None:
            filepath = self.league_dir / opponent.filepath
            if filepath.exists():
                checkpoint = torch.load(filepath, map_location="cpu")
                opponent._model_state_dict = checkpoint["model_state_dict"]
            else:
                raise FileNotFoundError(f"Opponent checkpoint not found: {filepath}")

    def get_opponent_state_dict(self, opponent: LeagueOpponent) -> Dict:
        """Get model state dict for an opponent."""
        self._ensure_loaded(opponent)
        return opponent._model_state_dict

    def load_from_disk(self) -> bool:
        """Load existing league from disk.

        Returns:
            True if league was loaded, False if no existing league found
        """
        if not self.metadata_path.exists():
            return False

        with open(self.metadata_path, "r") as f:
            data = json.load(f)

        self.opponents = [
            LeagueOpponent.from_dict(opp_data)
            for opp_data in data.get("opponents", [])
        ]

        # Validate that checkpoint files exist
        valid_opponents = []
        for opp in self.opponents:
            filepath = self.league_dir / opp.filepath
            if filepath.exists():
                valid_opponents.append(opp)
            else:
                print(f"Warning: League opponent checkpoint not found: {filepath}")

        self.opponents = valid_opponents
        return True

    def save_metadata(self):
        """Save league metadata to disk."""
        data = {
            "opponents": [opp.to_dict() for opp in self.opponents],
            "max_opponents": self.max_opponents,
        }

        with open(self.metadata_path, "w") as f:
            json.dump(data, f, indent=2)

    def __len__(self) -> int:
        """Return number of opponents in pool."""
        return len(self.opponents)

    def __repr__(self) -> str:
        """String representation."""
        if not self.opponents:
            return "League(empty)"

        win_rates = [opp.win_rate for opp in self.opponents]
        return (
            f"League({len(self.opponents)} opponents, "
            f"win_rates={min(win_rates):.2f}-{max(win_rates):.2f})"
        )
