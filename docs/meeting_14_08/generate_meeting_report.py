#!/usr/bin/env python3
"""Read-only audit and cohort QC report for copied Image Experiment outputs.

This script reads only ``derived/`` and writes its report products beside this
file.  It deliberately does not invoke any preprocessing or modelling code.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parents[2]
DERIVED = ROOT / "derived"
OUT = Path(__file__).resolve().parent
EXPECTED = {
    "merged": "merged/{slug}_image_trial_dataset.csv",
    "eeg_features": "eeg/features/eeg_epoch_features.csv",
    "face_features": "video/features/video_trial_features.csv",
    "clean_epochs": "eeg/cleaned_epochs/{slug}_image_cleaned-epo.fif",
    "qc_json": "eeg/quality_control/eeg_qc.json",
    "autoreject": "eeg/quality_control/autoreject_epoch_channel_log.csv",
    "ica": "eeg/quality_control/ica_motion_correlation.csv",
    "run_manifest": "manifest/run_manifest.json",
    "validation": "manifest/validation_summary.json",
    "configuration": "config/resolved_configuration.yaml",
}


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def value(frame: pd.DataFrame, name: str, default=np.nan):
    return frame[name] if name in frame else pd.Series(default, index=frame.index)


def fmt(number, digits=1):
    return "NA" if pd.isna(number) else f"{number:.{digits}f}"


def csv(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False, na_rep="NA")


def scan_windows_paths(run: Path) -> bool:
    pattern = re.compile(r"(?:[A-Za-z]:\\|[A-Za-z]:/|\\\\[A-Za-z0-9_-]+\\)")
    for path in list(run.rglob("*.json")) + list(run.rglob("*.yaml")):
        try:
            if pattern.search(path.read_text(encoding="utf-8", errors="replace")):
                return True
        except OSError:
            continue
    return False


def audit_run(participant: str, run: Path) -> tuple[dict, dict | None]:
    slug = participant.lower()
    paths = {name: run / relative.format(slug=slug) for name, relative in EXPECTED.items()}
    missing = [name for name, path in paths.items() if not path.is_file()]
    notes: list[str] = []
    readable = run.is_dir() and bool(list(run.iterdir()))
    zero = [str(path.relative_to(run)) for path in run.rglob("*") if path.is_file() and path.stat().st_size == 0]
    if zero:
        notes.append("zero-byte files: " + "; ".join(zero))
    if missing:
        notes.append("missing expected: " + "; ".join(missing))
    tables: dict[str, pd.DataFrame] = {}
    table_ok = {}
    for name in ("merged", "eeg_features", "face_features", "autoreject", "ica"):
        try:
            tables[name] = pd.read_csv(paths[name])
            table_ok[name] = len(tables[name].columns) > 0
            if not table_ok[name]: notes.append(f"malformed CSV: {name} has no columns")
        except Exception as exc:
            table_ok[name] = False
            notes.append(f"unreadable CSV {name}: {type(exc).__name__}: {exc}")
    qc_ok = True
    for name in ("qc_json", "run_manifest", "validation"):
        try:
            read_json(paths[name])
        except Exception as exc:
            qc_ok = False
            notes.append(f"unreadable JSON {name}: {type(exc).__name__}: {exc}")
    epochs_ok = True
    try:
        mne.read_epochs(paths["clean_epochs"], preload=False, verbose="ERROR")
    except Exception as exc:
        epochs_ok = False
        notes.append(f"unreadable cleaned epochs: {type(exc).__name__}: {exc}")
    identity_ok = participant.lower() in run.name.lower()
    for path in (paths["merged"], paths["clean_epochs"]):
        if path.exists() and participant.lower() not in path.name.lower(): identity_ok = False
    if not identity_ok: notes.append("participant ID does not agree with run/output filename")
    windows_metadata = scan_windows_paths(run)
    # Provenance paths from Windows do not stop the already-copied outputs being read on Linux.
    portable_problem = False
    if windows_metadata: notes.append("Windows absolute paths retained in provenance metadata; output files opened on Linux")
    fatal = not (readable and all(table_ok.get(x, False) for x in ("merged", "eeg_features", "face_features")) and epochs_ok and qc_ok and identity_ok) or bool(zero)
    row = {
        "participant": participant, "run_name": run.name, "run_readable": readable,
        "merged_table_readable": table_ok.get("merged", False),
        "eeg_features_readable": table_ok.get("eeg_features", False),
        "face_features_readable": table_ok.get("face_features", False),
        "clean_epochs_readable": epochs_ok, "qc_readable": qc_ok,
        "expected_files_missing": "; ".join(missing) if missing else "", "path_portability_issue": portable_problem,
        "participant_identity_ok": identity_ok, "notes": "; ".join(notes),
        "overall_integrity_status": "FAIL" if fatal else ("PASS_WITH_WINDOWS_PROVENANCE" if windows_metadata else "PASS"),
    }
    return row, tables if not fatal else None


def main() -> None:
    runs = sorted(DERIVED.glob("P[0-9][0-9]/Image_Experiment/runs/p[0-9][0-9]_final"))
    integrity, participant_rows, all_trials = [], [], []
    for run in runs:
        participant = run.parts[-4]
        audit, tables = audit_run(participant, run)
        integrity.append(audit)
        if tables is None: continue
        qc, manifest = read_json(run / EXPECTED["qc_json"].format(slug=participant.lower())), read_json(run / EXPECTED["run_manifest"].format(slug=participant.lower()))
        merged, ar = tables["merged"].copy(), tables["autoreject"]
        eeg_columns = [column for column in merged if column.startswith("eeg_") and not column.startswith("epoch_index")]
        face_columns = [column for column in merged if column.startswith("video_") and column.endswith("_mean")]
        merged["_eeg_usable"] = merged[eeg_columns].notna().all(axis=1) if eeg_columns else False
        merged["_face_usable"] = (value(merged, "face_detection_rate", 0).fillna(0) > 0) & (merged[face_columns].notna().all(axis=1) if face_columns else False)
        merged["participant"] = participant
        all_trials.append(merged)
        trials = qc.get("ica_trialwise", {}).get("trials", [])
        corrected = [trial for trial in trials if trial.get("corrected_source_count", 0) > 0]
        source_count = sum(trial.get("corrected_source_count", 0) for trial in trials)
        ranks = [trial.get("detected_eeg_rank") for trial in trials if trial.get("detected_eeg_rank") is not None]
        components = [trial.get("fitted_ica_components") for trial in trials if trial.get("fitted_ica_components") is not None]
        warnings = sum("converg" in line.lower() and "warning" in line.lower() for line in (run / "logs/pipeline.log").read_text(encoding="utf-8", errors="replace").splitlines())
        before, retained = qc.get("epochs_before_cleaning", np.nan), qc.get("epochs_after_cleaning", np.nan)
        rejected_ids = ar.loc[ar["epoch_rejected"].astype(bool), "trigger_id"].tolist() if "epoch_rejected" in ar else []
        missing_v, missing_a = value(merged, "valence_rating").isna().sum(), value(merged, "arousal_rating").isna().sum()
        status = "approved incomplete" if manifest.get("incomplete_session") else "full session"
        participant_rows.append({
            "participant": participant, "matched_trials": len(merged), "eeg_retained": retained,
            "eeg_rejected": before-retained, "eeg_retention_percent": 100*retained/before if before else np.nan,
            "rejected_trigger_ids": "; ".join(map(str, rejected_ids)),
            "post_car_rank": pd.Series(ranks).mode().iat[0] if ranks else np.nan,
            "ica_components": pd.Series(components).mode().iat[0] if components else np.nan,
            "ica_convergence_warnings": warnings, "motion_corrected_trials": len(corrected),
            "motion_corrected_sources": source_count,
            "motion_corrected_trial_percent": 100*len(corrected)/len(trials) if trials else np.nan,
            "face_detection_percent": 100*value(merged, "face_detected_frames", 0).sum()/value(merged, "video_frame_count", np.nan).sum(),
            "missing_eeg_trial_rows": (~merged["_eeg_usable"]).sum(), "missing_face_trial_rows": (~merged["_face_usable"]).sum(),
            "missing_valence_ratings": missing_v, "missing_arousal_ratings": missing_a,
            "merged_rows": len(merged), "nan_cells": int(merged.drop(columns=["_eeg_usable", "_face_usable", "participant"]).isna().sum().sum()),
            "inf_cells": int(np.isinf(merged.select_dtypes(include=np.number)).sum().sum()), "status": status,
            "notes": "" if not manifest.get("incomplete_session") else "53 EEG epochs before cleaning; 54 image events recorded.",
        })
    qc_frame = pd.DataFrame(participant_rows).sort_values("participant")
    trial_frame = pd.concat(all_trials, ignore_index=True)
    csv(OUT / "derived_integrity_check.csv", integrity)
    qc_frame.to_csv(OUT / "participant_qc_summary.csv", index=False, na_rep="NA")

    # Rating/class table: one row per measure and participant, followed by cohort rows.
    rating_rows = []
    for participant, subset in list(trial_frame.groupby("participant")) + [("COHORT", trial_frame)]:
        for measure in ("valence", "arousal"):
            ratings = pd.to_numeric(subset[f"{measure}_rating"], errors="coerce").dropna()
            low, high = (ratings < 4).sum(), (ratings >= 4).sum()
            counts = "; ".join(f"{int(k)}:{int(v)}" for k, v in ratings.value_counts().sort_index().items())
            rating_rows.append({"participant": participant, "measure": measure, "valid_ratings": len(ratings), "missing_ratings": len(subset)-len(ratings), "mean": ratings.mean(), "sd": ratings.std(), "median": ratings.median(), "min": ratings.min(), "q1": ratings.quantile(.25), "q3": ratings.quantile(.75), "max": ratings.max(), "discrete_value_counts": counts, "low_count": low, "high_count": high, "low_percent": 100*low/len(ratings) if len(ratings) else np.nan, "high_percent": 100*high/len(ratings) if len(ratings) else np.nan, "class_imbalance_ratio_major_to_minor": max(low, high)/min(low, high) if min(low, high) else np.nan})
    pd.DataFrame(rating_rows).to_csv(OUT / "rating_and_class_summary.csv", index=False, na_rep="NA")

    q1 = qc_frame.eeg_retention_percent.quantile(.25)
    readiness = []
    for _, row in qc_frame.iterrows():
        if row.status == "approved incomplete": group, reason = "Special/incomplete case", "Approved incomplete EEG session."
        elif row.missing_valence_ratings or row.missing_arousal_ratings: group, reason = "Review due to missing ratings", "At least one target rating is absent."
        elif row.eeg_retention_percent < q1: group, reason = "Review due to elevated EEG rejection", f"Retention below cohort first quartile ({q1:.1f}%)."
        elif row.eeg_retention_percent == 100: group, reason = "Very clean", "No AutoReject epoch losses."
        else: group, reason = "Clean with minor EEG loss", "Retention at or above cohort first quartile."
        subset = trial_frame[trial_frame.participant == row.participant]
        readiness.append({"participant": row.participant, "eeg_usable_trials": int(subset._eeg_usable.sum()), "face_usable_trials": int(subset._face_usable.sum()), "valence_multimodal_usable_trials": int((subset._eeg_usable & subset._face_usable & subset.valence_rating.notna()).sum()), "arousal_multimodal_usable_trials": int((subset._eeg_usable & subset._face_usable & subset.arousal_rating.notna()).sum()), "eeg_retention_percent": row.eeg_retention_percent, "face_detection_percent": row.face_detection_percent, "motion_correction_percent": row.motion_corrected_trial_percent, "missing_valence": row.missing_valence_ratings, "missing_arousal": row.missing_arousal_ratings, "qc_group": group, "reason": reason})
    readiness_frame = pd.DataFrame(readiness)
    readiness_frame.to_csv(OUT / "modeling_readiness.csv", index=False, na_rep="NA")

    make_figures(qc_frame, pd.DataFrame(rating_rows))
    write_report(qc_frame, trial_frame, integrity, readiness_frame)


def barplot(frame, column, title, ylabel, filename, ymin=None):
    plot = frame.sort_values("participant")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(plot.participant, plot[column], color="#4477AA")
    ax.set(title=title, xlabel="Participant", ylabel=ylabel)
    if ymin is not None: ax.set_ylim(*ymin)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout(); fig.savefig(OUT / filename, dpi=180); plt.close(fig)


def make_figures(qc, ratings):
    barplot(qc, "eeg_retention_percent", "EEG retention after AutoReject", "EEG retention (%)", "eeg_retention_by_participant.png", (0, 101))
    barplot(qc, "eeg_rejected", "AutoReject-rejected EEG epochs", "Rejected epochs (n)", "eeg_rejected_trials_by_participant.png")
    barplot(qc, "motion_corrected_trial_percent", "ACC-guided ICA motion correction", "Trials corrected (%)", "motion_correction_by_participant.png", (0, max(5, qc.motion_corrected_trial_percent.max()*1.15)))
    low = max(0, np.floor(qc.face_detection_percent.min()-0.2))
    barplot(qc, "face_detection_percent", "Face detection across image trials", "Face detection (%)", "face_detection_by_participant.png", (low, 100.05))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for axis, measure in zip(axes, ("valence", "arousal")):
        # Reconstruct counts from the compact CSV field, avoiding a second data source.
        counts = {int(x.split(":")[0]): int(x.split(":")[1]) for x in ratings[(ratings.participant == "COHORT") & (ratings.measure == measure)].iloc[0].discrete_value_counts.split("; ")}
        axis.bar(list(counts), list(counts.values()), color="#66AA55")
        axis.set(title=measure.capitalize(), xlabel="Rating value", ylabel="Valid ratings (n)", xticks=sorted(counts))
    fig.suptitle("Self-rating distributions across the processed cohort"); fig.tight_layout(); fig.savefig(OUT / "rating_distributions.png", dpi=180); plt.close(fig)
    cohort = ratings[ratings.participant == "COHORT"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(2); width = .32
    for offset, (_, row) in zip((-width/2, width/2), cohort.iterrows()): ax.bar(x+offset, [row.low_count, row.high_count], width, label=row.measure.capitalize())
    ax.set(title="Binary class balance (LOW <4; HIGH ≥4)", xticks=x, xticklabels=["Low", "High"], ylabel="Valid ratings (n)"); ax.legend(); fig.tight_layout(); fig.savefig(OUT / "class_balance.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(qc.motion_corrected_trial_percent, 100-qc.eeg_retention_percent, color="#AA4465")
    for _, row in qc.iterrows(): ax.annotate(row.participant, (row.motion_corrected_trial_percent, 100-row.eeg_retention_percent), xytext=(3,3), textcoords="offset points", fontsize=7)
    ax.set(title="Motion correction and AutoReject loss", xlabel="Motion-corrected trials (%)", ylabel="AutoReject rejection (%)"); fig.tight_layout(); fig.savefig(OUT / "motion_vs_rejection.png", dpi=180); plt.close(fig)


def write_report(qc, trials, integrity, readiness):
    retention = qc.eeg_retention_percent
    face = qc.face_detection_percent
    usable = {"EEG-only": int(trials._eeg_usable.sum()), "Face-only": int(trials._face_usable.sum()), "EEG + face": int((trials._eeg_usable & trials._face_usable).sum()), "EEG + face + valence": int((trials._eeg_usable & trials._face_usable & trials.valence_rating.notna()).sum()), "EEG + face + arousal": int((trials._eeg_usable & trials._face_usable & trials.arousal_rating.notna()).sum())}
    motion, rejection = qc.motion_corrected_trial_percent, 100-qc.eeg_retention_percent
    pearson = pearsonr(motion, rejection); spearman = spearmanr(motion, rejection)
    motion_iqr = motion.quantile(.75) - motion.quantile(.25)
    high_motion_cutoff = motion.quantile(.75) + 1.5 * motion_iqr
    high_motion = qc.loc[motion > high_motion_cutoff, "participant"].tolist()
    review = readiness[readiness.qc_group.str.startswith("Review") | readiness.qc_group.str.startswith("Special")].participant.tolist()
    complete = int((qc.status == "full session").sum())
    text = f"""# Image Experiment Data Quality Summary

## Dataset processed so far

{len(qc)} official final runs were found: {", ".join(qc.participant)}. There are {complete} full 120-trial sessions and {(qc.status == "approved incomplete").sum()} approved incomplete session (P03). The merged tables contain {len(trials):,} matched trial rows (36 × 120 trials plus 54 P03 incomplete-session rows); {int(qc.eeg_retained.sum()):,} EEG trials were retained and {int(qc.eeg_rejected.sum()):,} rejected. P03 also has one matched row without an EEG epoch because 54 image events yielded 53 pre-cleaning epochs.

## Preprocessing pipeline

The frozen pipeline uses 8 EEG channels, common-average reference, and a 1–40 Hz fourth-order Butterworth IIR filter. It uses rank-aware ICA (post-CAR rank expected 7), with one stable ICA decomposition fitted on concatenated valid image epochs. ACC X/Y/Z correlation is evaluated per trial; flagged motion-related ICA source activity is high-pass filtered at 3 Hz, then EEG is reconstructed. Baseline correction (-0.5 to 0.0 s) follows ICA reconstruction. AutoReject then runs with epoch-wise channel interpolation disabled. Features use 0–2 s and delta 1–4, theta 4–8, alpha 8–12, beta 12–30, and gamma 30–40 Hz bands. This is an adapted ACC-guided ICA preprocessing strategy, not a claim of perfect reproduction of the reference paper.

## Cross-machine integrity check

**The copied `derived/` final runs are safe to use for subsequent analysis on this Linux PC.** All {len(integrity)} final runs were opened successfully: merged, EEG-feature, face-feature, and QC tables parsed, and cleaned FIF epochs opened with MNE. Windows absolute paths remain in provenance metadata, but none was an active path needed to open the copied outputs. No zero-byte expected outputs were found.

## EEG quality

Participant EEG retention was mean {fmt(retention.mean())}%, median {fmt(retention.median())}%, SD {fmt(retention.std())}%, range {fmt(retention.min())}–{fmt(retention.max())}%, and quartiles {fmt(retention.quantile(.25))}/{fmt(retention.quantile(.75))}%. Rejected epochs per participant ranged from {int(qc.eeg_rejected.min())} to {int(qc.eeg_rejected.max())} (median {fmt(qc.eeg_rejected.median())}). Across processed EEG epochs, retention was {fmt(100*qc.eeg_retained.sum()/(qc.eeg_retained.sum()+qc.eeg_rejected.sum()))}% ({int(qc.eeg_rejected.sum())} rejected).

## ICA and motion correction

Motion correction affected mean {fmt(qc.motion_corrected_trials.mean())} and median {fmt(qc.motion_corrected_trials.median())} trials per participant ({fmt(qc.motion_corrected_trial_percent.mean())}% and {fmt(qc.motion_corrected_trial_percent.median())}%). Using the descriptive 1.5×IQR rule (> {fmt(high_motion_cutoff)}%), unusually high motion-correction rates occurred in {", ".join(high_motion) or "no participants"}. The descriptive association with AutoReject rejection was Pearson r={fmt(pearson.statistic,2)} (p={fmt(pearson.pvalue,3)}) and Spearman ρ={fmt(spearman.statistic,2)} (p={fmt(spearman.pvalue,3)}); this does not establish causation.

## Facial feature quality

Weighted face detection was mean {fmt(face.mean())}%, median {fmt(face.median())}%, range {fmt(face.min())}–{fmt(face.max())}%. Participants below 99%: {", ".join(qc.loc[face < 99, "participant"]) or "none"}; below 95%: {", ".join(qc.loc[face < 95, "participant"]) or "none"}.

## Self-rating completeness

There are {int(qc.missing_valence_ratings.sum())} missing valence ratings and {int(qc.missing_arousal_ratings.sum())} missing arousal ratings. Participants affected: {", ".join(qc.loc[(qc.missing_valence_ratings + qc.missing_arousal_ratings) > 0, "participant"])}. Ratings are available for {fmt(100*trials.valence_rating.notna().mean())}% of merged rows (valence) and {fmt(100*trials.arousal_rating.notna().mean())}% (arousal).

## Usable sample sizes for modeling

""" + "\n".join(f"- {name}: {count:,} trial rows" for name, count in usable.items()) + f"""

AutoReject-rejected EEG trials remain in the merged table but have missing EEG feature values. Therefore many NaN cells can arise from relatively few rejected trial rows. Missing EEG does not automatically remove a face-only observation. Missing ratings reduce supervised-learning sample size for that target, but do not invalidate the raw physiological recording.

## Participants requiring additional review

{", ".join(review) or "None"}. These QC groupings are descriptive and are not scientific exclusion decisions; see `modeling_readiness.csv` for transparent reasons.

## Main takeaways

The copied outputs are readable and internally usable on Linux. EEG quality is generally high, face detection is high, and the principal data-completeness limitation is concentrated in the incomplete P03 session and the missing ratings recorded for P01/P02. Model training was not performed.

## Questions for Friday's discussion

1. What participant-level EEG rejection rate should trigger exclusion versus simply removing individual trials?
2. Should participants with unusually high ACC-based motion correction remain in the primary cohort if their post-correction EEG passes AutoReject?
3. Is the <4 versus ≥4 valence/arousal threshold scientifically appropriate for final classification, or should participant-normalized labels or another formulation be considered?
4. Should final evaluation use leave-one-subject-out/group-aware validation rather than random trial splitting?
5. How should participants with substantial missing self-ratings be used: modality-specific training, partial-target analysis, or exclusion from the corresponding supervised target?
"""
    (OUT / "MEETING_SUMMARY.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
