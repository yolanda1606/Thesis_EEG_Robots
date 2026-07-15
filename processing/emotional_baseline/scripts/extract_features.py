#!/usr/bin/env python
import argparse
from pathlib import Path
import sys

import mne
import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from emotional_baseline.config import OUTPUT_DIR
from emotional_baseline.features import extract_epoch_features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--participant")
    args = parser.parse_args()
    pattern = f"{args.participant}/segment-*-epo.fif" if args.participant else "*/segment-*-epo.fif"
    frames = []
    for path in sorted((OUTPUT_DIR / "epochs").glob(pattern)):
        epochs = mne.read_epochs(path, preload=True, verbose="ERROR")
        frames.append(extract_epoch_features(epochs, path.parent.name, path.stem))
    if not frames:
        raise SystemExit("No preprocessed epoch files found")
    output = OUTPUT_DIR / "features" / "eeg_features.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(output, index=False)
    print(f"Saved features to {output}")


if __name__ == "__main__":
    main()
