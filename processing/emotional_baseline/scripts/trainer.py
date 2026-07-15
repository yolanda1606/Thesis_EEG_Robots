#!/usr/bin/env python
import argparse
from pathlib import Path
import sys

import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from emotional_baseline.config import OUTPUT_DIR
from emotional_baseline.trainer import train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=["valence", "arousal"], required=True)
    args = parser.parse_args()
    features = pd.read_csv(OUTPUT_DIR / "features" / "labeled_eeg_features.csv")
    destination = OUTPUT_DIR / "models" / f"{args.target}_baseline.joblib"
    destination.parent.mkdir(parents=True, exist_ok=True)
    summary = train(features, args.target, destination)
    print(summary)
    print(f"Saved model to {destination}")


if __name__ == "__main__":
    main()
