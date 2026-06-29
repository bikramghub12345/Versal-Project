#!/usr/bin/env python3
# MBU Fault Injection - Plot Results
# Target : BUFFERS
# Run    : python3 plot_results.py

import os, glob, pandas as pd, matplotlib.pyplot as plt
import matplotlib.ticker as mticker

OUTDIR    = os.path.dirname(os.path.abspath(__file__))
CSV_DIR   = os.path.join(OUTDIR, 'csv')
PLOTS_DIR = os.path.join(OUTDIR, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

TARGET = 'BUFFERS'

# Chart 1: Accuracy vs Bits Flipped
acc_path = os.path.join(OUTDIR, 'accuracy_summary.csv')
if os.path.exists(acc_path):
    df_acc   = pd.read_csv(acc_path)
    base_acc = df_acc['baseline_accuracy_pct'].iloc[0]
    x_labels = ['0 (baseline)'] + df_acc['bits'].astype(str).tolist()
    acc_vals = [base_acc] + df_acc['accuracy_pct'].tolist()
    colors   = ['forestgreen'] + ['steelblue'] * len(df_acc)
    fig_w    = max(10, len(x_labels) * 0.9)
    fig, ax  = plt.subplots(figsize=(fig_w, 6))
    bars = ax.bar(x_labels, acc_vals, color=colors,
                  edgecolor='black', width=0.65)
    ax.axhline(base_acc, color='forestgreen', linestyle='--', linewidth=1.5,
               label='Baseline %.1f%%' % base_acc)
    ax.set_xlabel('Bits Flipped (k)', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('MBU Fault Injection -- ' + TARGET + '\nAccuracy vs Bit Count',
                 fontsize=13, fontweight='bold')
    ax.set_ylim(0, 110)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.legend(fontsize=10)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    for bar, v in zip(bars, acc_vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 1,
                '%.1f%%' % v, ha='center', va='bottom', fontsize=7)
    plt.tight_layout()
    out_path = os.path.join(OUTDIR, 'plot_accuracy_vs_bits.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print('[Plot] Saved: ' + out_path)

# Chart 2: Prob Drop per Image (one chart per k, saved to plots/ subfolder)
csv_files = sorted(glob.glob(os.path.join(CSV_DIR, 'results_k*_bits.csv')))
for csv_path in csv_files:
    k_str = os.path.basename(csv_path).replace('results_k', '').replace('_bits.csv', '')
    try:
        k_val = int(k_str)
    except ValueError:
        continue
    df = pd.read_csv(csv_path)
    df = df[df['crash'] == 0].copy()
    if df.empty:
        continue
    avg   = df.groupby('image_name')['prob_drop'].mean().reset_index()
    avg   = avg.sort_values('image_name')
    short = [os.path.basename(n) for n in avg['image_name']]
    fig_w = max(10, len(short) * 0.45)
    fig, ax = plt.subplots(figsize=(fig_w, 5))
    colors  = ['tomato' if v > 0.05 else 'steelblue' for v in avg['prob_drop']]
    ax.bar(short, avg['prob_drop'], color=colors, edgecolor='black')
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xlabel('Image', fontsize=11)
    ax.set_ylabel('Avg Probability Drop', fontsize=11)
    ax.set_title('MBU -- ' + TARGET + '  k=' + str(k_val) + ' bits\n'
                 'Probability Drop per Image  (red = drop > 0.05)',
                 fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=7)
    plt.tight_layout()
    fname    = 'plot_prob_drop_k' + str(k_val) + '.png'
    out_path = os.path.join(PLOTS_DIR, fname)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print('[Plot] Saved: ' + out_path)

print('[Done]')
