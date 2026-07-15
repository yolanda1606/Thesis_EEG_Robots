#!/usr/bin/env python
from pathlib import Path
import sys

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from emotional_baseline.config import OUTPUT_DIR
from emotional_baseline.inventory import build_inventory, write_inventory


def main() -> None:
    output = OUTPUT_DIR / "inventory" / "image_experiment_inventory.json"
    inventory = build_inventory()
    write_inventory(inventory, output)
    print(f"Wrote {len(inventory['records'])} participant records to {output}")


if __name__ == "__main__":
    main()
