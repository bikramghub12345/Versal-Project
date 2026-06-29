#!/usr/bin/env python3
import os, pandas as pd, matplotlib.pyplot as plt
import matplotlib.ticker as mticker

OUTDIR = os.path.dirname(os.path.abspath(__file__))

acc_path = os.path.join(OUTDIR, 'accuracy_summary.csv')
if os.path.exists(acc_path):
    df = pd.read_csv(acc_path)
    base_acc = df['baseline_accuracy_pct'].iloc[0]
    x = ['0 (baseline)'] + df['bits'].astype(str).tolist()
    y = [base_acc] + df['accuracy_pct'].tolist()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x, y, color=['forestgreen']+['steelblue']*len(df), edgecolor='black')
    ax.axhline(base_acc, color='forestgreen', linestyle='--',
               label=f'Baseline {base_acc:.1f}%')
    ax.set_xlabel('Bits flipped (k)'); ax.set_ylabel('Accuracy')
    ax.set_title('MBU Fault Injection: Accuracy vs Bit Count')
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.legend()
    for i, v in enumerate(y): ax.text(i, v+1, f'{v:.1f}%', ha='center')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR,'plot_accuracy_vs_bits.png'), dpi=150)
    plt.close()
    print('[Plot] plot_accuracy_vs_bits.png')

for k in [100,1000,5000,10000,20000]:
    csv = os.path.join(OUTDIR, f'results_k{k}_bits.csv')
    if not os.path.exists(csv): continue
    df = pd.read_csv(csv)
    df = df[(df.timeout==0)&(df.crash==0)]
    avg = df.groupby('image_name')['prob_drop'].mean().reset_index()
    fig, ax = plt.subplots(figsize=(max(8,len(avg)*0.8), 5))
    colors = ['tomato' if v>0.05 else 'steelblue' for v in avg.prob_drop]
    ax.bar([n[-20:] for n in avg.image_name], avg.prob_drop,
           color=colors, edgecolor='black')
    ax.set_title(f'MBU k={k} bits: Prob Drop per Image')
    plt.xticks(rotation=45, ha='right'); plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR,f'plot_prob_drop_k{k}.png'),dpi=150)
    plt.close()
    print(f'[Plot] plot_prob_drop_k{k}.png')
