#!/usr/bin/env python3
"""
plot_all.py — Master SEFI Results Plotter
==========================================
Place this file in ./FaultResults/sefi_results/ and run:
    python3 plot_all.py

It walks every numbered mode folder (e.g. "01. SEFI-row") and every
target subfolder inside it (weights / input_tensor / buffers), reads
the CSVs, and saves three PNG plots per folder:
    plot_accuracy_<mode>_<target>.png
    plot_prob_drop_<mode>_<target>.png
    plot_bytes_corrupted_<mode>_<target>.png  (skipped if all zeros)

Requirements: matplotlib only (no pandas/numpy needed).
"""

import os
import csv
import matplotlib
matplotlib.use("Agg")          # non-interactive, works without a display
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from collections import defaultdict

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
SEFI_ROOT = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def read_csv(path):
    """Return list-of-dicts from a CSV file, or [] if missing."""
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def avg_by_image(rows, field):
    """Return (sorted_image_names, averages) grouping field by image_name."""
    bucket = defaultdict(list)
    for r in rows:
        try:
            bucket[r["image_name"]].append(float(r[field]))
        except (KeyError, ValueError):
            pass
    names = sorted(bucket.keys())
    avgs  = [sum(bucket[n]) / len(bucket[n]) for n in names]
    return names, avgs


def short_name(full_name, n=25):
    """Last n chars of image path for x-axis labels."""
    return full_name[-n:]


# ---------------------------------------------------------------------------
# PLOT FUNCTIONS
# ---------------------------------------------------------------------------
def plot_accuracy(acc_path, mode, target, out_dir):
    rows = read_csv(acc_path)
    if not rows:
        print(f"  [Skip] No accuracy_summary.csv in {out_dir}")
        return

    row        = rows[0]
    base_acc   = float(row.get("baseline_accuracy_pct", 0))
    faulty_acc = float(row.get("faulty_accuracy_pct",  0))
    recov_acc  = float(row.get("recovery_accuracy_pct", 0))

    # Include recovery bar only for MSEFI modes (recovery > 0 or msefi_recovered > 0)
    has_recovery = any(
        int(r.get("msefi_recovered", 0)) for r in rows
    ) or recov_acc > 0

    if has_recovery:
        labels = ["Baseline", "Post-SEFI", "Post-Recovery"]
        vals   = [base_acc, faulty_acc, recov_acc]
        colors = ["forestgreen", "tomato", "steelblue"]
    else:
        labels = ["Baseline", "Post-SEFI"]
        vals   = [base_acc, faulty_acc]
        colors = ["forestgreen", "tomato"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, vals, color=colors, edgecolor="black", width=0.5)
    ax.axhline(base_acc, color="forestgreen", linestyle="--",
               label=f"Baseline {base_acc:.1f}%")
    for i, v in enumerate(vals):
        ax.text(i, v + 1.5, f"{v:.1f}%", ha="center", fontsize=11)
    ax.set_ylim(0, 115)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_title(f"SEFI: {mode} | Target: {target}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()

    out_png = os.path.join(out_dir, f"plot_accuracy_{mode}_{target}.png")
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"  [Plot] {out_png}")


def plot_prob_drop(res_path, mode, target, out_dir):
    rows = [r for r in read_csv(res_path) if r.get("crash", "0") == "0"]
    if not rows:
        print(f"  [Skip] No valid rows for prob_drop in {out_dir}")
        return

    names, avgs = avg_by_image(rows, "prob_drop")
    if not names:
        return

    short = [short_name(n) for n in names]
    colors = ["tomato" if v > 0.05 else "steelblue" for v in avgs]

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.9), 5))
    ax.bar(short, avgs, color=colors, edgecolor="black")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title(f"Prob Drop | {mode} | {target}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Image")
    ax.set_ylabel("Baseline prob − Faulty prob")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.tight_layout()

    out_png = os.path.join(out_dir, f"plot_prob_drop_{mode}_{target}.png")
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"  [Plot] {out_png}")


def plot_bytes_corrupted(res_path, mode, target, out_dir):
    rows = [r for r in read_csv(res_path) if r.get("crash", "0") == "0"]
    if not rows or "bytes_corrupted" not in rows[0]:
        return

    names, avgs = avg_by_image(rows, "bytes_corrupted")
    if not names or max(avgs) == 0:
        return

    short = [short_name(n) for n in names]

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.9), 5))
    ax.bar(short, avgs, color="plum", edgecolor="black")
    ax.set_title(f"Bytes Corrupted | {mode} | {target}", fontsize=12, fontweight="bold")
    ax.set_ylabel("Bytes corrupted per image (avg)")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.tight_layout()

    out_png = os.path.join(out_dir, f"plot_bytes_corrupted_{mode}_{target}.png")
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"  [Plot] {out_png}")


# ---------------------------------------------------------------------------
# WALK sefi_results/ AND PLOT EVERYTHING
# ---------------------------------------------------------------------------
def main():
    print(f"[plot_all] Scanning: {SEFI_ROOT}\n")

    mode_folders = sorted(
        d for d in os.listdir(SEFI_ROOT)
        if os.path.isdir(os.path.join(SEFI_ROOT, d))
        and d[0].isdigit()           # numbered folders only: "01. SEFI-row" etc.
    )

    if not mode_folders:
        print("[plot_all] No numbered mode folders found. Nothing to plot.")
        return

    total_plots = 0

    for mode_folder in mode_folders:
        mode_path = os.path.join(SEFI_ROOT, mode_folder)
        # Strip leading "XX. " to get the clean mode name for plot titles/filenames
        mode_name = mode_folder.split(". ", 1)[-1] if ". " in mode_folder else mode_folder

        print(f"[Mode] {mode_folder}")

        target_folders = sorted(
            d for d in os.listdir(mode_path)
            if os.path.isdir(os.path.join(mode_path, d))
            and d not in (".", "..")
        )

        if not target_folders:
            print("  [Skip] No target subfolders found.")
            continue

        for target_name in target_folders:
            target_path = os.path.join(mode_path, target_name)

            acc_csv = os.path.join(target_path, "accuracy_summary.csv")
            # Find the results CSV (named results_<mode>.csv)
            res_csv = None
            for fn in os.listdir(target_path):
                if fn.startswith("results_") and fn.endswith(".csv"):
                    res_csv = os.path.join(target_path, fn)
                    break

            print(f"  [Target] {target_name}")

            if os.path.exists(acc_csv):
                plot_accuracy(acc_csv, mode_name, target_name, target_path)
                total_plots += 1
            else:
                print(f"    [Skip] accuracy_summary.csv not found")

            if res_csv:
                plot_prob_drop(res_csv, mode_name, target_name, target_path)
                plot_bytes_corrupted(res_csv, mode_name, target_name, target_path)
                total_plots += 2
            else:
                print(f"    [Skip] results_*.csv not found")

        print()

    print(f"[plot_all] Done. {total_plots} plots generated across {len(mode_folders)} mode(s).")


if __name__ == "__main__":
    main()
