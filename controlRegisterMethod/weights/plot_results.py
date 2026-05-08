#!/usr/bin/env python3
import os, pandas as pd, matplotlib.pyplot as plt
import matplotlib.ticker as mticker, numpy as np

OUTDIR = os.path.dirname(os.path.abspath(__file__))

acc_path = os.path.join(OUTDIR, 'accuracy_summary.csv')
if os.path.exists(acc_path):
    df_acc = pd.read_csv(acc_path)
    baseline_acc = df_acc['baseline_accuracy_pct'].iloc[0]
    x_labels = ['0 (baseline)'] + df_acc['bits'].astype(str).tolist()
    y_vals   = [baseline_acc]   + df_acc['accuracy_pct'].tolist()
    colors   = ['forestgreen'] + ['steelblue']*len(df_acc)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x_labels, y_vals, color=colors, edgecolor='black', width=0.5)
    ax.axhline(baseline_acc, color='forestgreen', linestyle='--',
               linewidth=1.2, label=f'Baseline {baseline_acc:.1f}%')
    ax.set_xlabel('Number of Flipped Bits (k)', fontsize=12)
    ax.set_ylabel('Accuracy (ground truth)', fontsize=12)
    ax.set_title('MBU Fault Injection: Accuracy vs Bit Count', fontsize=13)
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.legend(fontsize=10)
    for i, v in enumerate(y_vals):
        ax.text(i, v+1.5, f'{v:.1f}%', ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR,'plot_accuracy_vs_bits.png'),dpi=150)
    plt.close()
    print('[Plot] Saved: plot_accuracy_vs_bits.png')

bit_counts = [100,1000,5000,10000,20000]

for k in bit_counts:
    csv_path = os.path.join(OUTDIR, f'results_k{k}_bits.csv')
    if not os.path.exists(csv_path): continue
    df = pd.read_csv(csv_path)
    df_valid = df[(df['timeout']==0) & (df['crash']==0)].copy()
    if df_valid.empty: continue
    avg_drop = df_valid.groupby('image_name')['prob_drop'].mean().reset_index()
    avg_drop = avg_drop.sort_values('image_name')
    short_names = [os.path.splitext(n)[0][-20:] for n in avg_drop['image_name']]
    fig, ax = plt.subplots(figsize=(max(8, len(short_names)*0.8), 5))
    colors = ['tomato' if v > 0.05 else 'steelblue' for v in avg_drop['prob_drop']]
    ax.bar(short_names, avg_drop['prob_drop'], color=colors, edgecolor='black')
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xlabel('Image', fontsize=11)
    ax.set_ylabel('Avg Prob Drop', fontsize=10)
    ax.set_title(f'MBU k={k} bits: Probability Drop per Image', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, f'plot_prob_drop_k{k}.png'), dpi=150)
    plt.close()
    print(f'[Plot] Saved: plot_prob_drop_k{k}.png')

print('[Done]')
