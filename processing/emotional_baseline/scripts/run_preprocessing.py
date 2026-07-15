#!/usr/bin/env python
import argparse
import json
from pathlib import Path
import sys

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from emotional_baseline.config import OUTPUT_DIR
from emotional_baseline.preprocessing import preprocess_segment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--participant", required=True)
    args = parser.parse_args()
    manifest = OUTPUT_DIR / "inventory" / "image_experiment_inventory.json"
    inventory = json.loads(manifest.read_text(encoding="utf-8"))
    matches = [record for record in inventory["records"] if record["participant"] == args.participant]
    if len(matches) != 1:
        raise SystemExit(f"Expected one inventory record for {args.participant}; found {len(matches)}")
    output_dir = OUTPUT_DIR / "epochs" / args.participant
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, segment in enumerate(matches[0]["eeg_segments"], start=1):
        if "error" in segment:
            print(f"Skipping unreadable segment: {segment['error']}")
            continue
        epochs = preprocess_segment(Path(segment["path"]))
        destination = output_dir / f"segment-{index:02d}-epo.fif"
        epochs.save(destination, overwrite=False)
        print(f"Saved {len(epochs)} epochs to {destination}")


if __name__ == "__main__":
    main()
