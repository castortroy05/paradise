#!/usr/bin/env python3
"""
Archetype classifier module for P.A.R.A.D.I.S.E.
Calculates weighted archetype fit scores (0.0-1.0) and assigns primary/secondary roles.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from modules.normalization import ZScoreNormalizer


@dataclass(frozen=True)
class ArchetypeSpec:
    name: str
    description: str
    weights: Dict[str, float]


DEFAULT_ARCHETYPES: List[ArchetypeSpec] = [
    ArchetypeSpec(
        name="Progressive Pivot",
        description="High passing volume with progressive distribution.",
        weights={
            "passes": 0.35,
            "progressive_passes": 0.35,
            "pass_accuracy": 0.2,
            "progressive_carries": 0.1,
        },
    ),
    ArchetypeSpec(
        name="High-Press Predator",
        description="Aggressive pressing and ball-winning in advanced areas.",
        weights={
            "tackles": 0.35,
            "interceptions": 0.3,
            "pressing_actions": 0.25,
            "blocks": 0.1,
        },
    ),
    ArchetypeSpec(
        name="Creative Playmaker",
        description="Chance creation through key passes and shot creation.",
        weights={
            "key_passes": 0.4,
            "assists": 0.25,
            "shot_creating_actions": 0.25,
            "progressive_passes": 0.1,
        },
    ),
    ArchetypeSpec(
        name="Box-to-Box Warrior",
        description="Balanced two-way contributor across phases.",
        weights={
            "tackles": 0.25,
            "interceptions": 0.2,
            "progressive_carries": 0.25,
            "shots": 0.15,
            "aerial_duels": 0.15,
        },
    ),
    ArchetypeSpec(
        name="Deep-Lying Controller",
        description="Tempo setter with secure passing and long distribution.",
        weights={
            "passes": 0.3,
            "pass_accuracy": 0.3,
            "long_balls": 0.25,
            "progressive_passes": 0.15,
        },
    ),
    ArchetypeSpec(
        name="Goal-Scoring Threat",
        description="Primary finisher with high shot and goal output.",
        weights={
            "shots": 0.35,
            "goals": 0.35,
            "xg": 0.2,
            "touches_in_box": 0.1,
        },
    ),
    ArchetypeSpec(
        name="Defensive Anchor",
        description="Defensive stopper who wins duels and clears danger.",
        weights={
            "tackles": 0.3,
            "interceptions": 0.3,
            "clearances": 0.25,
            "aerial_duels": 0.15,
        },
    ),
    ArchetypeSpec(
        name="Dynamic Dribbler",
        description="Progressive carrier who beats defenders on the dribble.",
        weights={
            "dribbles": 0.35,
            "take_ons": 0.25,
            "progressive_carries": 0.3,
            "touches_in_box": 0.1,
        },
    ),
]


SAMPLE_DATA = {
    "name": [
        "De Bruyne",
        "Rodri",
        "Casemiro",
        "Bruno",
        "Haaland",
        "Kane",
        "Salah",
        "Saka",
        "Van Dijk",
        "Dias",
    ],
    "position": ["CM", "CDM", "CDM", "CAM", "ST", "ST", "RW", "RW", "CB", "CB"],
    "league": [
        "Premier League",
        "Premier League",
        "Premier League",
        "Premier League",
        "Premier League",
        "Premier League",
        "Premier League",
        "Premier League",
        "Premier League",
        "Premier League",
    ],
    "tackles": [40, 95, 88, 35, 20, 25, 30, 45, 80, 75],
    "interceptions": [35, 80, 75, 30, 15, 20, 25, 40, 90, 85],
    "passes": [2200, 2500, 1800, 1900, 600, 900, 1200, 1400, 1800, 2000],
    "pass_accuracy": [88.5, 91.2, 87.3, 85.6, 72.4, 78.9, 82.1, 84.3, 89.7, 90.1],
    "progressive_passes": [180, 140, 100, 160, 50, 70, 90, 110, 130, 120],
    "key_passes": [85, 40, 25, 90, 20, 45, 70, 60, 15, 12],
    "shots": [60, 20, 15, 80, 150, 120, 140, 90, 10, 8],
    "goals": [8, 2, 1, 10, 35, 28, 25, 15, 2, 1],
    "dribbles": [70, 25, 20, 80, 40, 35, 110, 95, 15, 12],
    "progressive_carries": [95, 60, 45, 100, 55, 50, 120, 105, 35, 30],
    "assists": [16, 4, 3, 18, 6, 10, 12, 11, 2, 1],
    "shot_creating_actions": [130, 80, 55, 140, 70, 95, 120, 110, 35, 32],
    "pressing_actions": [200, 280, 260, 220, 120, 140, 180, 190, 90, 85],
    "blocks": [20, 35, 40, 15, 10, 12, 14, 16, 55, 60],
    "aerial_duels": [30, 60, 70, 25, 55, 65, 40, 35, 90, 85],
    "long_balls": [120, 150, 110, 90, 40, 70, 60, 80, 100, 110],
    "xg": [10.2, 3.1, 1.8, 12.4, 32.6, 26.5, 22.8, 14.7, 1.5, 1.2],
    "touches_in_box": [50, 25, 18, 55, 160, 130, 140, 100, 12, 10],
}


class ArchetypeClassifier:
    """Compute archetype fit scores and assignments using weighted metrics."""

    def __init__(
        self,
        archetypes: Optional[Iterable[ArchetypeSpec]] = None,
        normalize: str = "none",
        league_column: str = "league",
        position_column: str = "position",
    ) -> None:
        self.archetypes = list(archetypes) if archetypes else DEFAULT_ARCHETYPES
        self.normalize = normalize
        self.league_column = league_column
        self.position_column = position_column

    def score_players(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return dataframe with archetype scores and primary/secondary assignments."""
        if data.empty:
            raise ValueError("Input data is empty.")

        data = data.copy()
        metrics = self._collect_metrics()
        normalized_data = self._apply_normalization(data, metrics)
        score_matrix = self._build_score_matrix(normalized_data, metrics)

        for archetype_name, scores in score_matrix.items():
            data[score_column_name(archetype_name)] = scores

        primary, secondary = self._assign_top_two(score_matrix)
        data["primary_archetype"] = primary["name"]
        data["primary_score"] = primary["score"]
        data["secondary_archetype"] = secondary["name"]
        data["secondary_score"] = secondary["score"]

        return data

    def _collect_metrics(self) -> List[str]:
        metrics = set()
        for archetype in self.archetypes:
            metrics.update(archetype.weights.keys())
        return sorted(metrics)

    def _apply_normalization(self, data: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
        if self.normalize == "none":
            return data

        normalizer = ZScoreNormalizer()
        if self.normalize == "global":
            return normalizer.normalize(data, metrics, self.league_column)
        if self.normalize == "position":
            return normalizer.normalize_by_position(
                data,
                metrics,
                position_column=self.position_column,
                league_column=self.league_column,
            )
        raise ValueError(f"Unsupported normalize option: {self.normalize}")

    def _build_score_matrix(
        self,
        data: pd.DataFrame,
        metrics: List[str],
    ) -> Dict[str, np.ndarray]:
        metric_values = self._prepare_metric_values(data, metrics)
        score_matrix: Dict[str, np.ndarray] = {}

        for archetype in self.archetypes:
            raw_score = np.zeros(len(data), dtype=float)
            for metric, weight in archetype.weights.items():
                raw_score += weight * metric_values[metric]
            score_matrix[archetype.name] = min_max_scale(raw_score)

        return score_matrix

    def _prepare_metric_values(
        self,
        data: pd.DataFrame,
        metrics: List[str],
    ) -> Dict[str, np.ndarray]:
        metric_values: Dict[str, np.ndarray] = {}
        for metric in metrics:
            normalized_col = f"{metric}_norm"
            if normalized_col in data.columns:
                series = data[normalized_col]
            elif metric in data.columns:
                series = z_score(data[metric])
            else:
                series = pd.Series(0.0, index=data.index)
            metric_values[metric] = series.fillna(0.0).to_numpy(dtype=float)
        return metric_values

    def _assign_top_two(
        self,
        score_matrix: Dict[str, np.ndarray],
    ) -> Tuple[Dict[str, pd.Series], Dict[str, pd.Series]]:
        score_df = pd.DataFrame(score_matrix)
        score_values = score_df.to_numpy()
        top_indices = np.argsort(-score_values, axis=1)[:, :2]
        archetype_names = score_df.columns.tolist()

        primary_name = pd.Series(
            [archetype_names[idx] for idx in top_indices[:, 0]],
            index=score_df.index,
        )
        secondary_name = pd.Series(
            [archetype_names[idx] for idx in top_indices[:, 1]],
            index=score_df.index,
        )

        primary_score = pd.Series(
            score_values[np.arange(len(score_df)), top_indices[:, 0]],
            index=score_df.index,
        )
        secondary_score = pd.Series(
            score_values[np.arange(len(score_df)), top_indices[:, 1]],
            index=score_df.index,
        )

        primary = {"name": primary_name, "score": primary_score}
        secondary = {"name": secondary_name, "score": secondary_score}

        return primary, secondary


def z_score(series: pd.Series) -> pd.Series:
    std = series.std()
    if std == 0 or np.isnan(std):
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std


def min_max_scale(values: np.ndarray) -> np.ndarray:
    min_val = np.min(values)
    max_val = np.max(values)
    if np.isclose(max_val, min_val):
        return np.full_like(values, 0.5, dtype=float)
    return (values - min_val) / (max_val - min_val)


def score_column_name(archetype_name: str) -> str:
    return f"archetype_score_{slugify(archetype_name)}"


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9\s_-]", "", value)
    value = re.sub(r"[\s_-]+", "_", value)
    return value


def load_data(path: Optional[str], use_sample: bool) -> pd.DataFrame:
    if use_sample:
        return pd.DataFrame(SAMPLE_DATA)
    if not path:
        raise ValueError("Provide --input or use --sample to load data.")

    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    if input_path.suffix.lower() in {".json", ".jsonl"}:
        return pd.read_json(input_path, lines=input_path.suffix.lower() == ".jsonl")
    return pd.read_csv(input_path)


def export_data(data: pd.DataFrame, output_path: Optional[str]) -> None:
    if not output_path:
        return

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix.lower() == ".json":
        path.write_text(json.dumps(data.to_dict(orient="records"), indent=2))
    else:
        data.to_csv(path, index=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute archetype fit scores and assign primary/secondary roles.",
    )
    parser.add_argument("--input", "-i", help="Path to CSV/JSON player data.")
    parser.add_argument("--output", "-o", help="Path to write tagged players.")
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Use built-in sample data for quick testing.",
    )
    parser.add_argument(
        "--normalize",
        choices=["none", "global", "position"],
        default="none",
        help="Apply Z-score normalization before scoring.",
    )
    parser.add_argument(
        "--league-column",
        default="league",
        help="League column name (used when normalizing).",
    )
    parser.add_argument(
        "--position-column",
        default="position",
        help="Position column name (used for position normalization).",
    )
    parser.add_argument(
        "--show",
        type=int,
        default=10,
        help="Number of rows to display in the CLI output.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    data = load_data(args.input, args.sample)
    classifier = ArchetypeClassifier(
        normalize=args.normalize,
        league_column=args.league_column,
        position_column=args.position_column,
    )

    scored = classifier.score_players(data)

    display_columns = [
        col
        for col in [
            "name",
            "position",
            "primary_archetype",
            "primary_score",
            "secondary_archetype",
            "secondary_score",
        ]
        if col in scored.columns
    ]

    print("\nArchetype assignments (top results):")
    print(scored[display_columns].head(args.show).to_string(index=False))

    if "primary_archetype" in scored.columns:
        counts = scored["primary_archetype"].value_counts().to_frame("count")
        print("\nPrimary archetype distribution:")
        print(counts.to_string())

    export_data(scored, args.output)
    if args.output:
        print(f"\nTagged players exported to: {args.output}")


if __name__ == "__main__":
    main()
