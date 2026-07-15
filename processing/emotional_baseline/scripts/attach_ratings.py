#!/usr/bin/env python
from pathlib import Path
import sys

import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from emotional_baseline.config import OUTPUT_DIR
from emotional_baseline.labels import attach_ratings


def main() -> None:
    feature_path = OUTPUT_DIR / "features" / "eeg_features.csv"
    inventory_path = OUTPUT_DIR / "inventory" / "image_experiment_inventory.json"
    labeled = attach_ratings(pd.read_csv(feature_path), inventory_path)
    output = OUTPUT_DIR / "features" / "labeled_eeg_features.csv"
    labeled.to_csv(output, index=False)
    matched = int((labeled["rating_match"] == "both").sum())
    print(f"Saved {len(labeled)} labeled rows ({matched} matched) to {output}")


if __name__ == "__main__":
    main()
