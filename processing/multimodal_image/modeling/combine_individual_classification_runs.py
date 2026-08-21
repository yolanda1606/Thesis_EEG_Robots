#!/usr/bin/env python3
"""Combine compatible completed individual-classification runs without retraining."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))


SETTINGS = ("mode", "target", "modalities", "models", "feature_counts", "seed", "cv_strategy", "hyperparameter_grids", "label_definition")
TARGETS = ("valence", "arousal")
STRICT = {"P10", "P11", "P12", "P14", "P15", "P16", "P17", "P18", "P19", "P20", "P22", "P23", "P24", "P25", "P26", "P27", "P30", "P31", "P32", "P33", "P34", "P38", "P39", "P40", "P41", "P42", "P43", "P44", "P45", "P46"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-run", type=Path, required=True)
    parser.add_argument("--second-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def manifest(path: Path) -> dict:
    return json.loads((path / "run_manifest.json").read_text(encoding="utf-8"))


def best_table(summary: pd.DataFrame, parameters: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["participant", "target"]
    winners = summary[summary.balanced_accuracy_mean == summary.groupby(keys).balanced_accuracy_mean.transform("max")].copy()
    winners["feature_count_request"] = winners.feature_count_request.astype(str)
    order = {"all": 0, "5": 1, "10": 2, "20": 3}
    winners["_order"] = winners.feature_count_request.map(order)
    chosen = winners.sort_values(["participant", "target", "modality", "classifier", "_order"], kind="stable").groupby(keys, as_index=False).first()
    ties = winners.groupby(keys).size().rename("winner_tie_count")
    chosen = chosen.merge(ties, on=keys, validate="one_to_one")
    grouped = parameters.copy(); grouped.feature_count_request = grouped.feature_count_request.astype(str)
    frequency = grouped.groupby(["participant", "target", "modality", "classifier", "feature_count_request", "best_parameters"]).size().rename("count").reset_index()
    frequent = frequency.sort_values("count", ascending=False, kind="stable").groupby(["participant", "target", "modality", "classifier", "feature_count_request"], as_index=False).first()
    frequent["most_frequent_hyperparameters"] = frequent.apply(lambda row: f"{row.best_parameters} ({row['count']}/5 folds)", axis=1)
    return chosen.merge(frequent.drop(columns=["best_parameters", "count"]), on=["participant", "target", "modality", "classifier", "feature_count_request"], validate="one_to_one"), winners


def save_figures(best: pd.DataFrame, winners: pd.DataFrame, out: Path) -> None:
    participants = sorted(best.participant.unique(), key=lambda p: int(p[1:])); x = np.arange(len(participants)); width = .38
    fig, ax = plt.subplots(figsize=(18, 6))
    for offset, target, color in ((-width/2, "valence", "#4477AA"), (width/2, "arousal", "#CC6677")):
        values = best[best.target == target].set_index("participant").loc[participants, "balanced_accuracy_mean"]
        ax.bar(x+offset, values, width, label=target.capitalize(), color=color)
    ax.axhline(.5, color="black", ls="--", label="Chance (BA = 0.50)"); ax.set(xticks=x, xticklabels=participants, ylim=(0, 1), ylabel="Best nested-CV balanced accuracy", title="Combined completed runs: best individual performance (P01–P46)")
    ax.legend(ncol=3, frameon=False); ax.grid(axis="y", alpha=.25); fig.tight_layout(); fig.savefig(out / "combined_best_ba_per_participant.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 6)); values = [best[best.target == target].balanced_accuracy_mean for target in TARGETS]
    boxes = ax.boxplot(values, tick_labels=[target.capitalize() for target in TARGETS], patch_artist=True)
    for box, color in zip(boxes["boxes"], ("#4477AA", "#CC6677")): box.set_facecolor(color); box.set_alpha(.45)
    for index, value in enumerate(values, 1): ax.scatter(np.full(len(value), index) + np.linspace(-.1, .1, len(value)), value, color="#333333", s=15, zorder=3)
    ax.axhline(.5, color="black", ls="--"); ax.set(ylim=(0, 1), ylabel="Best nested-CV balanced accuracy", title="Distribution of participant-best BA")
    ax.grid(axis="y", alpha=.25); fig.tight_layout(); fig.savefig(out / "combined_best_ba_distribution.png", dpi=180); plt.close(fig)
    for column, labels, filename, title in (("modality", ["eeg", "face", "multimodal", "Tie"], "combined_winning_modality_counts.png", "Winning modality"), ("classifier", ["knn", "svm", "gnb", "Tie"], "combined_winning_classifier_counts.png", "Winning classifier")):
        rows=[]
        for target in TARGETS:
            for participant, group in winners[winners.target == target].groupby("participant"):
                choices=set(group[column]); rows.append({"target":target,"choice":next(iter(choices)) if len(choices)==1 else "Tie"})
        counts=pd.DataFrame(rows).groupby(["target","choice"]).size().unstack(fill_value=0).reindex(columns=labels, fill_value=0)
        fig, ax=plt.subplots(figsize=(8,5)); positions=np.arange(len(labels))
        for offset,target,color in ((-.18,"valence","#4477AA"),(.18,"arousal","#CC6677")):
            bars=ax.bar(positions+offset, counts.loc[target], .36, label=target.capitalize(), color=color); ax.bar_label(bars,padding=2)
        ax.set(xticks=positions,xticklabels=[label.upper() if label!='Tie' else label for label in labels],ylim=(0,46),ylabel="Participants (n = 46)",title=title); ax.legend(frameon=False); ax.grid(axis="y",alpha=.25); fig.tight_layout(); fig.savefig(out/filename,dpi=180); plt.close(fig)
    thresholds=(.55,.60,.65); labels=("≥ 0.55","≥ 0.60","≥ 0.65"); fig,ax=plt.subplots(figsize=(7,5)); positions=np.arange(3)
    for offset,target,color in ((-.18,"valence","#4477AA"),(.18,"arousal","#CC6677")):
        values=best[best.target==target].balanced_accuracy_mean; counts=[int((values>=threshold).sum()) for threshold in thresholds]; bars=ax.bar(positions+offset,counts,.36,label=target.capitalize(),color=color); ax.bar_label(bars,labels=[f"{n}/46" for n in counts],padding=2)
    ax.set(xticks=positions,xticklabels=labels,ylim=(0,48),ylabel="Participants (n = 46)",title="Cumulative participant-best BA thresholds"); ax.legend(frameon=False); ax.grid(axis="y",alpha=.25); fig.tight_layout(); fig.savefig(out/"combined_participants_above_threshold.png",dpi=180); plt.close(fig)


def main() -> None:
    args = parse_args(); first, second = manifest(args.first_run), manifest(args.second_run)
    if {key:first.get(key) for key in SETTINGS} != {key:second.get(key) for key in SETTINGS}: raise ValueError("Source runs have different scientific settings; refusing to combine")
    source = [(args.first_run, first), (args.second_run, second)]
    participant_sets = [set(item[1]["resolved_participants"]) for item in source]
    expected={f"P{number:02d}" for number in range(1,47)}
    if participant_sets[0] & participant_sets[1] or participant_sets[0] | participant_sets[1] != expected: raise ValueError("Source-run participants are not a disjoint P01-P46 partition")
    if args.output_dir.exists(): raise FileExistsError(f"Refusing to overwrite existing combined analysis directory: {args.output_dir}")
    summaries=[]; parameters=[]
    for path, run_manifest in source:
        summary=pd.read_csv(path/"results_summary.csv"); parameter=pd.read_csv(path/"best_hyperparameters.csv")
        if set(summary.participant) != set(run_manifest["resolved_participants"]): raise ValueError(f"Summary participant mismatch in {path}")
        summary["source_run"]=path.name; summaries.append(summary); parameters.append(parameter)
    summary=pd.concat(summaries,ignore_index=True); parameter=pd.concat(parameters,ignore_index=True)
    if summary.duplicated(["participant","target","modality","classifier","feature_count_request"]).any(): raise ValueError("Duplicate configurations found")
    best,winners=best_table(summary,parameter); best["qc_cohort"] = np.where(best.participant.isin(STRICT), "Strict QC-approved", "Technically usable; QC caveat")
    args.output_dir.mkdir(parents=True)
    columns=["participant","target","qc_cohort","source_run","balanced_accuracy_mean","accuracy_mean","precision_mean","recall_mean","f1_mean","modality","classifier","feature_count_request","balanced_accuracy_sd","balanced_accuracy_median","balanced_accuracy_min","balanced_accuracy_max","most_frequent_hyperparameters","winner_tie_count"]
    best[columns].sort_values(["participant","target"]).to_csv(args.output_dir/"combined_best_per_participant.csv",index=False)
    save_figures(best,winners,args.output_dir)
    caveat=sorted(set(best.participant)-STRICT,key=lambda p:int(p[1:])); lines=["# Combined Individual Classification: P01–P46", "", "## Provenance", "", f"This directory combines completed summaries from `{args.first_run}` and `{args.second_run}`. It is not a single training invocation and contains no retraining output.", "", "The source manifests match exactly for individual mode, targets, modalities, classifiers, feature-count settings, nested StratifiedKFold CV logic, random seed (42), hyperparameter grids, and label definition. The source participant sets are disjoint and together cover P01–P46 exactly once.", "", "## QC strata", "", f"- Strict QC-approved participants (30): {', '.join(sorted(STRICT,key=lambda p:int(p[1:])))}", f"- Technically usable but QC-caveat participants (16): {', '.join(caveat)}", "", "## Participant-best balanced accuracy", ""]
    for target in TARGETS:
        values=best[best.target==target].balanced_accuracy_mean; lines.append(f"- {target.capitalize()}: mean {values.mean():.3f}; median {values.median():.3f}; SD {values.std():.3f}; range {values.min():.3f}–{values.max():.3f}; ≥0.55 {(values>=.55).sum()}/46; ≥0.60 {(values>=.60).sum()}/46; ≥0.65 {(values>=.65).sum()}/46.")
    lines += ["", "Participant-best figures select the highest completed nested-CV configuration for each participant and target. They are descriptive; selecting among many configurations is not an independent estimate of a pre-specified deployed model."]
    (args.output_dir/"COMBINED_ALL46_SUMMARY.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


if __name__ == "__main__": main()
