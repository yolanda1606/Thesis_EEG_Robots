#!/usr/bin/env python3
"""Read-only comparability checks for P19 canonical and no-ICA Image runs."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_RUN = Path("derived/P19/Image_Experiment/runs/p19_final")
NO_ICA_RUN = Path("derived/P19/Image_Experiment/runs/p19_no_ica")
RATING_COLUMNS = ("trigger", "stim_id", "category", "valence_rating", "arousal_rating")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-run", type=Path, default=CANONICAL_RUN)
    parser.add_argument("--no-ica-run", type=Path, default=NO_ICA_RUN)
    return parser.parse_args(argv)


def read_required(run: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    features_path = run / "eeg/features/eeg_epoch_features.csv"
    merged_path = run / "merged/p19_image_trial_dataset.csv"
    if not features_path.is_file() or not merged_path.is_file():
        raise FileNotFoundError(f"Expected EEG feature and merged tables under: {run}")
    return pd.read_csv(features_path), pd.read_csv(merged_path)


def compare_runs(canonical_run: Path, no_ica_run: Path) -> tuple[bool, list[str]]:
    """Check identifiers/schema exactly; numeric EEG values may differ by design."""
    canonical_features, canonical_merged = read_required(canonical_run)
    no_ica_features, no_ica_merged = read_required(no_ica_run)
    messages: list[str] = []
    required_feature = {"epoch_index", "trigger", "channel"}
    for name, frame in (("canonical", canonical_features), ("no_ica", no_ica_features)):
        if missing := sorted(required_feature.difference(frame.columns)):
            raise ValueError(f"{name} feature table missing required columns: {missing}")
    canonical_eeg = [column for column in canonical_features if column.startswith("eeg_")]
    no_ica_eeg = [column for column in no_ica_features if column.startswith("eeg_")]
    schema_match = canonical_eeg == no_ica_eeg
    channel_match = set(canonical_features.channel) == set(no_ica_features.channel)
    canonical_keys = set(map(tuple, canonical_features[["trigger", "channel"]].to_numpy()))
    no_ica_keys = set(map(tuple, no_ica_features[["trigger", "channel"]].to_numpy()))
    trial_match = canonical_keys == no_ica_keys
    messages.extend([
        f"Canonical retained EEG rows: {len(canonical_features)}; no-ICA retained EEG rows: {len(no_ica_features)}.",
        f"Canonical retained triggers: {canonical_features.trigger.nunique()}; no-ICA retained triggers: {no_ica_features.trigger.nunique()}.",
        f"EEG feature column names identical: {schema_match}; count canonical/no-ICA: {len(canonical_eeg)}/{len(no_ica_eeg)}.",
        f"Channel set identical: {channel_match}; participant/trial keys identical: {trial_match}.",
    ])
    missing_ratings = [column for column in RATING_COLUMNS if column not in canonical_merged or column not in no_ica_merged]
    if missing_ratings:
        raise ValueError(f"Merged tables lack rating/identifier columns: {missing_ratings}")
    canonical_ratings = canonical_merged[list(RATING_COLUMNS)].sort_values("trigger").reset_index(drop=True)
    no_ica_ratings = no_ica_merged[list(RATING_COLUMNS)].sort_values("trigger").reset_index(drop=True)
    ratings_match = canonical_ratings.equals(no_ica_ratings)
    messages.append(f"Target/rating rows identical: {ratings_match}; merged row count canonical/no-ICA: {len(canonical_merged)}/{len(no_ica_merged)}.")
    metadata_columns = [column for column in canonical_features.columns if not column.startswith("eeg_")]
    metadata_match = (set(metadata_columns) == set(column for column in no_ica_features.columns if not column.startswith("eeg_")) and
                      canonical_features[metadata_columns].sort_values(["trigger", "channel"]).reset_index(drop=True).equals(
                          no_ica_features[metadata_columns].sort_values(["trigger", "channel"]).reset_index(drop=True)))
    messages.append(f"Non-feature EEG metadata identical: {metadata_match}. Numeric EEG feature values are intentionally permitted to differ because ICA is disabled.")
    return schema_match and channel_match and trial_match and ratings_match and metadata_match, messages


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    passed, messages = compare_runs(args.canonical_run, args.no_ica_run)
    print("P19 no-ICA comparability check: " + ("PASS" if passed else "INCLUSION_OR_SCHEMA_DIFFERENCE"))
    print("\n".join(messages))
    return 0 if passed else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
