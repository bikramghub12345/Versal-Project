#!/usr/bin/env python3
# SEFI Simulation Plots — mode: MSEFI-DDR-pwrcycle, target: weights

import os, pandas as pd, matplotlib.pyplot as plt
import matplotlib.ticker as mticker

OUTDIR = os.path.dirname(os.path.abspath(__file__))

acc_path = os.path.join(OUTDIR, 'accuracy_summary.csv')
if os.path.exists(acc_path):
    df  = pd.read_csv(acc_path)
    row = df.iloc[0]
    labels = ['Baseline', 'Post-SEFI', 'Post-Recovery']
    vals   = [row.baseline_accuracy_pct, row.faulty_accuracy_pct, row.recovery_accuracy_pct]
    colors = ['forestgreen', 'tomato', 'steelblue']
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, vals, color=colors, edgecolor='black', width=0.5)
    ax.axhline(row.baseline_accuracy_pct, color='forestgreen',
               linestyle='--', label=f'Baseline {row.baseline_accuracy_pct:.1f}%')
    for i, v in enumerate(vals): ax.text(i, v+1.5, f'{v:.1f}%', ha='center', fontsize=11)
    ax.set_ylim(0, 115)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_title('SEFI Fault Injection — Accuracy Impact', fontsize=14, fontweight='bold')
    ax.text(0.02, 0.98, f'Mode: {mode_name}\nTarget: {target_name}', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.8))
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'plot_accuracy_MSEFI-DDR-pwrcycle_weights.png'), dpi=150)
    plt.close(); print('[Plot] plot_accuracy_MSEFI-DDR-pwrcycle_weights.png')

csv_path = os.path.join(OUTDIR, 'results_MSEFI-DDR-pwrcycle.csv')
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    df = df[(df.timeout == 0) & (df.crash == 0)]
    avg = df.groupby('image_name')['prob_drop'].mean().reset_index()
    short = [n[-25:] for n in avg.image_name]
    colors = ['tomato' if v > 0.05 else 'steelblue' for v in avg.prob_drop]
    fig, ax = plt.subplots(figsize=(max(8, len(avg)*0.9), 5))
    ax.bar(short, avg.prob_drop, color=colors, edgecolor='black')
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_title('SEFI Fault Injection — Probability Drop per Image', fontsize=14, fontweight='bold')
    ax.text(0.02, 0.98, f'Mode: {mode_name}\nTarget: {target_name}', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.8))
    ax.set_xlabel('Image'); ax.set_ylabel('Baseline prob − Faulty prob')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'plot_prob_drop_MSEFI-DDR-pwrcycle_weights.png'), dpi=150)
    plt.close(); print('[Plot] plot_prob_drop_MSEFI-DDR-pwrcycle_weights.png')

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    df = df[(df.timeout == 0) & (df.crash == 0)]
    if 'bytes_corrupted' in df.columns and df.bytes_corrupted.max() > 0:
        avg2 = df.groupby('image_name')['bytes_corrupted'].mean().reset_index()
        fig, ax = plt.subplots(figsize=(max(8, len(avg2)*0.9), 5))
        ax.bar([n[-25:] for n in avg2.image_name], avg2.bytes_corrupted,
               color='plum', edgecolor='black')
        ax.set_title('SEFI Fault Injection — Bytes Corrupted per Image', fontsize=14, fontweight='bold')
        ax.text(0.02, 0.98, f'Mode: {mode_name}\nTarget: {target_name}', 
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.8))
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, 'plot_bytes_corrupted_MSEFI-DDR-pwrcycle_weights.png'), dpi=150)
        plt.close(); print('[Plot] plot_bytes_corrupted_MSEFI-DDR-pwrcycle_weights.png')

print('[Done] Plots saved to:', OUTDIR)
