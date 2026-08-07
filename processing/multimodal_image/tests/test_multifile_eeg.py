"""Synthetic tests for ordered EEG fragments and incomplete sessions."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import mne
import numpy as np
import pandas as pd

from processing.multimodal_image.run_image_pipeline import input_manifest, merge_tables
from processing.multimodal_image.src.configuration import _validate_eeg_input_choice
from processing.multimodal_image.src.eeg_sources import load_eeg_session
from processing.multimodal_image.src.validation import (available_image_codes,
                                                        resolve_experiment_directory,
                                                        resolve_inputs)


NAMES = [f"EEG {index}" for index in range(1, 9)] + ["ACC X", "ACC Y", "ACC Z", "GYR X", "GYR Y", "GYR Z", "Status"]
CONFIG = {
    "channels": {"eeg_mapping": {f"EEG {index}": name for index, name in enumerate(("Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"), 1)},
                 "motion_channels": ["ACC X", "ACC Y", "ACC Z", "GYR X", "GYR Y", "GYR Z"], "stim_channel": "Status"},
    "events": {"image_ranges": {"HAHV": [101, 130], "HALV": [201, 230], "LAHV": [301, 330], "LALV": [401, 430]}},
}


def synthetic_raw(triggers, *, sfreq=250.0, names=NAMES, samples=1000):
    types = ["eeg"] * 8 + ["misc"] * 6 + ["stim"]
    info = mne.create_info(names, sfreq, types[:len(names)])
    data = np.zeros((len(names), samples))
    status = names.index("Status")
    for offset, trigger in enumerate(triggers, 1):
        data[status, 100 * offset] = trigger
    return mne.io.RawArray(data, info, verbose="ERROR")


class TestMultiFileEeg(unittest.TestCase):
    def test_experiment_directory_resolves_both_supported_spellings(self):
        with tempfile.TemporaryDirectory() as directory:
            participant = Path(directory) / "participant"
            underscored = participant / "Image_Experiment"
            underscored.mkdir(parents=True)
            inputs = {"participant_dir": "participant", "experiment_dir": "Image Experiment"}
            self.assertEqual(resolve_experiment_directory(inputs, Path(directory)), underscored)

            spaced = participant / "Image Experiment"
            spaced.mkdir()
            self.assertEqual(resolve_experiment_directory(inputs, Path(directory)), spaced)

    def test_single_and_multiple_choice(self):
        _validate_eeg_input_choice({"inputs": {"eeg_file": "one.bdf"}})
        _validate_eeg_input_choice({"inputs": {"eeg_files": ["one.bdf", "two.bdf"]}})
        with self.assertRaisesRegex(ValueError, "exactly one"):
            _validate_eeg_input_choice({"inputs": {"eeg_file": "one.bdf", "eeg_files": ["two.bdf"]}})
        with self.assertRaisesRegex(ValueError, "non-empty ordered"):
            _validate_eeg_input_choice({"inputs": {"eeg_files": []}})

    def test_missing_fragment_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            experiment = Path(directory) / "participant" / "Image Experiment"
            experiment.mkdir(parents=True)
            for filename in ("ratings.csv", "video.avi", "log.csv"):
                (experiment / filename).touch()
            (Path(directory) / "model.task").touch()
            config = {"inputs": {"participant_dir": "participant", "experiment_dir": "Image Experiment",
                                  "eeg_files": ["EEG/missing.bdf"], "ratings_file": "ratings.csv",
                                  "video_file": "video.avi", "vision_log_file": "log.csv"},
                      "resources": {"face_landmarker_filename": "model.task"}}
            with self.assertRaisesRegex(FileNotFoundError, "EEG fragment"):
                resolve_inputs(config, Path(directory), Path(directory))

    def test_offsets_order_and_duplicate_rejection(self):
        raw1, raw2 = synthetic_raw([101, 102]), synthetic_raw([201])
        first_samples = raw1.n_times
        paths = {"eeg_files": [Path("first.bdf"), Path("second.bdf")]}
        with patch("processing.multimodal_image.src.eeg_sources.mne.io.read_raw_bdf", side_effect=[raw1, raw2]):
            combined, events, metadata = load_eeg_session(paths, CONFIG, preload=False)
        self.assertEqual(events[:, 2].tolist(), [101, 102, 201])
        self.assertEqual(events[2, 0], first_samples + 100)
        self.assertEqual(metadata[1]["cumulative_sample_offset"], first_samples)
        self.assertIn("BAD boundary", combined.annotations.description)
        with patch("processing.multimodal_image.src.eeg_sources.mne.io.read_raw_bdf",
                   side_effect=[synthetic_raw([101]), synthetic_raw([101])]):
            with self.assertRaisesRegex(ValueError, "across EEG fragments"):
                load_eeg_session(paths, CONFIG, preload=False)

    def test_channel_and_sampling_mismatch_fail(self):
        paths = {"eeg_files": [Path("first.bdf"), Path("second.bdf")]}
        changed = NAMES.copy(); changed[0] = "Different"
        with patch("processing.multimodal_image.src.eeg_sources.mne.io.read_raw_bdf",
                   side_effect=[synthetic_raw([101]), synthetic_raw([201], names=changed)]):
            with self.assertRaisesRegex(ValueError, "channel names/order"):
                load_eeg_session(paths, CONFIG, preload=False)
        with patch("processing.multimodal_image.src.eeg_sources.mne.io.read_raw_bdf",
                   side_effect=[synthetic_raw([101]), synthetic_raw([201], sfreq=200)]):
            with self.assertRaisesRegex(ValueError, "sampling frequency"):
                load_eeg_session(paths, CONFIG, preload=False)

    def test_incomplete_partitions_and_merge(self):
        planned = available_image_codes(CONFIG)
        missing = {102, 206, 213, 219, 311, 422}
        incomplete = {**CONFIG, "trials": {"allow_incomplete_image_trials": True,
                      "expected_available_triggers": sorted(planned - missing), "known_missing_triggers": sorted(missing)}}
        self.assertEqual(len(available_image_codes(incomplete)), 114)
        invalid = {**CONFIG, "trials": {"allow_incomplete_image_trials": True,
                   "expected_available_triggers": sorted(planned - missing), "known_missing_triggers": []}}
        with self.assertRaisesRegex(ValueError, "partition"):
            available_image_codes(invalid)
        with tempfile.TemporaryDirectory() as directory:
            ratings = Path(directory) / "ratings.csv"
            pd.DataFrame({"trigger_sent": sorted(planned), "stim_id": range(120), "category": "x",
                          "valence_rating": 1, "arousal_rating": 1, "valence_rt": 1, "arousal_rt": 1}).to_csv(ratings, index=False)
            merged = merge_tables(ratings, None, None, incomplete)
            self.assertEqual(len(merged), 114)
            self.assertEqual(set(merged["trigger"]), planned - missing)

    def test_manifest_hashes_every_fragment(self):
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory) / "one.bdf", Path(directory) / "two.bdf"
            first.write_bytes(b"one"); second.write_bytes(b"two")
            metadata = [{"source_index": 0, "source_sample_count": 10}, {"source_index": 1, "source_sample_count": 20}]
            with patch("processing.multimodal_image.run_image_pipeline.load_eeg_session", return_value=(None, None, metadata)):
                manifest = input_manifest({"eeg_files": [first, second]}, {"inputs": {}})
            self.assertEqual([row["source_index"] for row in manifest["eeg_files"]], [0, 1])
            self.assertTrue(all(len(row["sha256"]) == 64 for row in manifest["eeg_files"]))


if __name__ == "__main__":
    unittest.main()
