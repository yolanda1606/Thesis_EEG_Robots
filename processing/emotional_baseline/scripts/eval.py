#!/usr/bin/env python
import argparse
import json
from pathlib import Path
import sys

import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from emotional_baseline.config import OUTPUT_DIR
from emotional_baseline.evaluation import evaluate_loso


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=["valence", "arousal"], required=True)
    args = parser.parse_args()
    features = pd.read_csv(OUTPUT_DIR / "features" / "labeled_eeg_features.csv")
    metrics, predictions = evaluate_loso(features, args.target)
    output = OUTPUT_DIR / "evaluation"
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{args.target}_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    predictions.to_csv(output / f"{args.target}_predictions.csv", index=False)
    print(metrics)


if __name__ == "__main__":
    main()
