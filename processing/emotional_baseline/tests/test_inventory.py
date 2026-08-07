from pathlib import Path
import tempfile
import unittest

from emotional_baseline.inventory import build_inventory, choose_rating_export


def test_choose_rating_export_prefers_completed_ratings():
    exports = [
        {"path": "empty.csv", "rows": 120, "missing_columns": [],
         "missing_valence": 120, "missing_arousal": 120},
        {"path": "complete.csv", "rows": 120, "missing_columns": [],
         "missing_valence": 0, "missing_arousal": 0},
    ]
    assert choose_rating_export(exports)["path"] == "complete.csv"


class InventoryDirectoryTest(unittest.TestCase):
    def test_inventory_recognizes_spaced_and_underscored_image_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "P01_2026-01-01" / "Image Experiment").mkdir(parents=True)
            (root / "P02_2026-01-02" / "Image_Experiment").mkdir(parents=True)

            inventory = build_inventory(root)

            self.assertEqual(
                [record["participant"] for record in inventory["records"]],
                ["P01_2026-01-01", "P02_2026-01-02"],
            )
            self.assertEqual(
                {Path(record["experiment_directory"]).name for record in inventory["records"]},
                {"Image Experiment", "Image_Experiment"},
            )
