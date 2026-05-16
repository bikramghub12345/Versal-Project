#!/usr/bin/env python3
# SEFI Simulation Plots — mode: MSEFI-DDR-reset, target: buffers
# Stdlib only: no pandas/numpy required

import os, csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

OUTDIR      = os.path.dirname(os.path.abspath(__file__))
MODE_NAME   = 'MSEFI-DDR-reset'
TARGET_NAME = 'buffers'

def read_csv(path):
    if not os.path.exists(path): return []
    with open(path, newline='') as fh:
        return list(csv.DictReader(fh))

acc_rows = read_csv(os.path.join(OUTDIR, 'accuracy_summary.csv'))
if acc_rows:
    row = acc_rows[0]
    base_acc   = float(row['baseline_accuracy_pct'])
    faulty_acc = float(row['faulty_accuracy_pct'])
    recov_acc  = float(row['recovery_accuracy_pct'])
    labels = ['Baseline', 'Post-SEFI', 'Post-Recovery']
    vals   = [base_acc, faulty_acc, recov_acc]
    colors = ['forestgreen', 'tomato', 'steelblue']
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, vals, color=colors, edgecolor='black', width=0.5)
    ax.axhline(base_acc, color='forestgreen', linestyle='--',
               label='Baseline 0.0%%' % base_acc)
    for i, v in enumerate(vals):
        ax.text(i, v + 1.5, '0.0%%' % v, ha='center', fontsize=11)
    ax.set_ylim(0, 115)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_title('SEFI: %s | Target: %s' % (MODE_NAME, TARGET_NAME),
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    plt.tight_layout()
    out_png = os.path.join(OUTDIR, 'plot_accuracy_%s_%s.png' % (MODE_NAME, TARGET_NAME))
    plt.savefig(out_png, dpi=150); plt.close()
    print('[Plot]', out_png)

res_rows = read_csv(os.path.join(OUTDIR, 'results_MSEFI-DDR-reset.csv'))
res_rows = [r for r in res_rows if r.get('crash','0') == '0']
if res_rows:
    # Average prob_drop per image
    from collections import defaultdict
    drops = defaultdict(list)
    for r in res_rows:
        drops[r['image_name']].append(float(r['prob_drop']))
    names  = sorted(drops.keys())
    avgs   = [sum(drops[n])/len(drops[n]) for n in names]
    short  = [n[-25:] for n in names]
    colors = ['tomato' if v > 0.05 else 'steelblue' for v in avgs]
    fig, ax = plt.subplots(figsize=(max(8, len(names)*0.9), 5))
    ax.bar(short, avgs, color=colors, edgecolor='black')
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_title('Prob Drop | %s | %s' % (MODE_NAME, TARGET_NAME),
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Image'); ax.set_ylabel('Baseline prob - Faulty prob')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.tight_layout()
    out_png = os.path.join(OUTDIR, 'plot_prob_drop_%s_%s.png' % (MODE_NAME, TARGET_NAME))
    plt.savefig(out_png, dpi=150); plt.close()
    print('[Plot]', out_png)

if res_rows and 'bytes_corrupted' in res_rows[0]:
    bc = defaultdict(list)
    for r in res_rows:
        bc[r['image_name']].append(float(r['bytes_corrupted']))
    if any(v > 0 for vals in bc.values() for v in vals):
        names2 = sorted(bc.keys())
        avgs2  = [sum(bc[n])/len(bc[n]) for n in names2]
        fig, ax = plt.subplots(figsize=(max(8, len(names2)*0.9), 5))
        ax.bar([n[-25:] for n in names2], avgs2, color='plum', edgecolor='black')
        ax.set_title('Bytes Corrupted | %s | %s' % (MODE_NAME, TARGET_NAME),
                     fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.tight_layout()
        out_png = os.path.join(OUTDIR, 'plot_bytes_corrupted_%s_%s.png' % (MODE_NAME, TARGET_NAME))
        plt.savefig(out_png, dpi=150); plt.close()
        print('[Plot]', out_png)

print('[Done] Plots saved to:', OUTDIR)
