"""Synthetic checks for non-interactive EEG QC output helpers."""
from __future__ import annotations

import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import mne
import numpy as np
from matplotlib import image as mpimg
from autoreject import AutoReject
from autoreject.autoreject import RejectLog

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.features import _feature_window_epochs
from src.qc import (autoreject_rows, save_autoreject_plot,
                    save_event_related_band_power, save_fz_time_frequency,
                    validate_autoreject_event_alignment)
from run_image_pipeline import configure_logging, setup_run


CONFIG = {
    "eeg": {"epoch": {"tmin_s": -0.5, "tmax_s": 2.0, "baseline_s": [-0.5, 0.0]}},
    "features": {"eeg_feature_window_s": [0.0, 2.0], "bands_hz": {
        "delta": [1.0, 4.0], "theta": [4.0, 8.0], "alpha": [8.0, 12.0],
        "beta": [12.0, 30.0], "gamma": [30.0, 40.0]}},
    "events": {"image_ranges": {"HAHV": [101, 130], "HALV": [201, 230], "LAHV": [301, 330], "LALV": [401, 430]}},
}


class TestQcOutputs(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        codes = np.array([101, 102, 201, 202, 301, 302, 401])
        data = rng.normal(scale=1e-6, size=(len(codes), 3, 251))
        data[:, :, 50:] += 0.2e-6  # a small post-onset synthetic response
        names = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]
        data = np.repeat(data, len(names) // 3 + 1, axis=1)[:, :len(names)]
        info = mne.create_info(names, sfreq=100, ch_types="eeg")
        info.set_montage(mne.channels.make_standard_montage("standard_1020"), on_missing="raise")
        self.epochs = mne.EpochsArray(data, info, events=np.c_[np.arange(len(codes)), np.zeros(len(codes), int), codes], tmin=-0.5, verbose="ERROR")

    def test_outputs_and_alignment(self):
        labels = np.zeros((len(self.epochs), len(self.epochs.ch_names)), dtype=int)
        labels[1, 0], labels[3, 1] = 2, 1
        bad_epochs = np.array([False, False, False, True, False, False, False])
        reject_log = RejectLog(bad_epochs, labels, self.epochs.ch_names)
        retained = self.epochs[np.flatnonzero(~bad_epochs)]
        validate_autoreject_event_alignment(self.epochs.events, retained.events, reject_log)
        rows = autoreject_rows(reject_log, self.epochs.events)
        self.assertEqual(rows[3]["trigger_id"], 202)
        self.assertTrue(rows[3]["epoch_rejected"])
        self.assertEqual(rows[1]["interpolated_channel_names"], "Fz")
        self.assertEqual(rows[3]["bad_channel_names"], "C3")
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            save_autoreject_plot(reject_log, out / "autoreject.png", "SYN", "test")
            missing = save_event_related_band_power(retained, CONFIG, out / "spectral.png", out / "spectral.csv", "SYN", "test")
            self.assertEqual(missing, [])
            self.assertGreater((out / "autoreject.png").stat().st_size, 0)
            height, width = mpimg.imread(out / "autoreject.png").shape[:2]
            self.assertGreater(width, height)
            self.assertGreater((out / "spectral.png").stat().st_size, 0)
            self.assertGreater((out / "spectral.csv").stat().st_size, 0)
            self.assertEqual((out / "spectral.csv").read_text(encoding="utf-8").count("HAHV"), 5 * 251)
            save_fz_time_frequency(retained, CONFIG, out / "tfr_fz.png", "SYN", "test", __import__("logging").getLogger())
            self.assertGreater((out / "tfr_fz.png").stat().st_size, 0)
        self.assertEqual((self.epochs.tmin, self.epochs.tmax), (-0.5, 2.0))
        features = _feature_window_epochs(self.epochs, CONFIG)
        self.assertEqual((features.tmin, features.tmax), (0.0, 2.0))

    def test_missing_category_is_reported(self):
        selected = self.epochs[:6]
        with tempfile.TemporaryDirectory() as directory:
            missing = save_event_related_band_power(selected, CONFIG, Path(directory) / "spectral.png", Path(directory) / "spectral.csv", "SYN", "missing")
        self.assertEqual(missing, ["LALV"])

    def test_autoreject_zero_interpolation_rejects_bad_epochs(self):
        """AutoReject 0.4.3 supports [0] and does not create label-2 repairs."""
        rng = np.random.default_rng(2)
        data = rng.normal(0, 1e-6, (16, 8, 251))
        data[0, 0, 80:140] = 1e-3
        info = self.epochs.info.copy()
        events = np.c_[np.arange(16), np.zeros(16, dtype=int), np.arange(101, 117)]
        source = mne.EpochsArray(data, info, events=events, tmin=-0.5, verbose="ERROR")
        estimator = AutoReject(n_interpolate=[0], cv=2, random_state=42, n_jobs=1, verbose=False)
        cleaned, log = estimator.fit_transform(source, return_log=True)
        self.assertTrue(all(value == 0 for value in estimator.n_interpolate_.values()))
        self.assertFalse(np.any(log.labels == 2))
        self.assertGreater(np.sum(log.bad_epochs), 0)
        self.assertLess(len(cleaned), len(source))
        validate_autoreject_event_alignment(source.events, cleaned.events, log)
        self.assertTrue(all(row["interpolation_count"] == 0 for row in autoreject_rows(log, source.events)))

    def test_run_setup_is_lazy(self):
        """Run setup creates only its actual config/manifest/log outputs."""
        config = {"output": {"log_filename": "pipeline.log"}}
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "P00" / "Image_Experiment" / "runs" / "synthetic"
            setup_run(run_dir, config, ["synthetic"], Namespace(), health_only=False)
            logger = configure_logging(run_dir, config, "INFO")
            logger.info("synthetic log")
            for handler in list(logger.handlers):
                handler.close()
            empty = [path for path in run_dir.rglob("*") if path.is_dir() and not any(path.iterdir())]
            self.assertEqual(empty, [])
            self.assertFalse((run_dir / "eeg").exists())
            self.assertFalse((run_dir / "video").exists())


if __name__ == "__main__":
    unittest.main()
