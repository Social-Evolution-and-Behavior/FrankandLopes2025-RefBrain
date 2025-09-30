#!/usr/bin/env python3
"""Headless runner to compute Fig3E/F plots using SWC skeletons.
Produces the same PNG/EPS files in the figures/ directory.
"""
import os
import glob
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind


def load_swc_coordinates(file_path):
    import pandas as _pd
    try:
        df = _pd.read_csv(file_path, sep=r'\s+', comment='#', header=None, engine='python')
        if df.shape[1] < 5:
            data = np.loadtxt(file_path)
            coords = data[:, 2:5]
        else:
            coords = df.iloc[:, 2:5].to_numpy(dtype=float)
        return coords
    except Exception as e:
        print(f'WARNING: failed to load SWC {file_path}: {e}')
        return np.empty((0, 3))


def average_bidirectional_distance(A, B):
    A = np.array(A)
    B = np.array(B)
    if A.size == 0 or B.size == 0:
        return float('nan')
    dists = cdist(A, B)
    avg_a_to_b = np.mean(np.min(dists, axis=1))
    avg_b_to_a = np.mean(np.min(dists, axis=0))
    return (avg_a_to_b + avg_b_to_a) / 2


def compute_pairwise_distances(swc_folder):
    swc_files = sorted(glob.glob(os.path.join(swc_folder, '*.swc')))
    print(f'INFO: compute_pairwise_distances - folder={swc_folder} files_found={len(swc_files)}')
    pairwise_means = []
    for i, file_i in enumerate(swc_files):
        coords_i = load_swc_coordinates(file_i)
        if coords_i.size == 0:
            print(f'  SKIP empty or unreadable: {file_i}')
            continue
        for file_j in swc_files[i+1:]:
            coords_j = load_swc_coordinates(file_j)
            if coords_j.size == 0:
                print(f'  SKIP empty or unreadable: {file_j}')
                continue
            val = average_bidirectional_distance(coords_i, coords_j)
            if not np.isnan(val):
                pairwise_means.append(val)
    print(f'INFO: computed {len(pairwise_means)} pairwise means for folder {swc_folder}')
    return pairwise_means


def plot_group(df, out_png, out_eps, colors, ylimit):
    if df['pairwise_mean'].dropna().size == 0:
        print(f'No data in df for plot {out_png} — skipping')
        return
    np.random.seed(321)
    df['x_jitter'] = df['Registration'].map({
        'affine+rigid': lambda: 0 + np.random.uniform(-0.15, 0.15),
        'diffeomorphic': lambda: 1 + np.random.uniform(-0.15, 0.15)
    }).apply(lambda f: f())
    means = df.groupby('Registration')['pairwise_mean'].mean()
    medians = df.groupby('Registration')['pairwise_mean'].median()
    stds = df.groupby('Registration')['pairwise_mean'].std()
    tstat, pval = ttest_ind(
        df[df['Registration']=='affine+rigid']['pairwise_mean'],
        df[df['Registration']=='diffeomorphic']['pairwise_mean'],
        equal_var=False
    )
    plt.figure(figsize=(6,6))
    for i, reg in enumerate(['affine+rigid','diffeomorphic']):
        subset = df[df['Registration']==reg]
        if subset.empty:
            print(f'No data for {reg} in {out_png} — skipping group')
            continue
        plt.scatter(subset['x_jitter'], subset['pairwise_mean'], facecolors=colors[reg], edgecolors='black', linewidth=1.5, alpha=0.8, s=80)
        med_val = medians.get(reg, np.nan) if hasattr(medians,'get') else (medians[reg] if reg in medians.index else np.nan)
        std_val = stds.get(reg, np.nan) if hasattr(stds,'get') else (stds[reg] if reg in stds.index else np.nan)
        if not np.isnan(med_val):
            plt.hlines(med_val, xmin=i-0.2, xmax=i+0.2, colors='black', linewidth=5)
        if not np.isnan(std_val):
            plt.errorbar(x=i, y=med_val, yerr=std_val, fmt='none', ecolor='black', linewidth=3, capsize=0)
    max_val = df['pairwise_mean'].max()
    y_annotation = max_val + (0.05 * max_val)
    plt.plot([0,1],[y_annotation]*2, color='black', linewidth=1)
    plt.text(0.5, y_annotation + 0.02*max_val, f"p = {pval:.3g}" if pval>=0.001 else "p < 0.001", ha='center', va='bottom', fontsize=12)
    plt.xticks([0,1], ['affine+rigid','diffeomorphic'])
    plt.xlim(-0.5,1.5)
    plt.ylim(0, ylimit)
    plt.ylabel('Mean Pairwise Distance (units)')
    sns.despine(top=True, right=True)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, format='png', dpi=600)
    plt.savefig(out_eps, format='eps', dpi=300)
    plt.close()
    print(f'WROTE: {out_png}, {out_eps}')


if __name__ == '__main__':
    # GABA
    left_GABA_affine_folder = os.path.join('data','skeletons','skeletons_affine+rigid','GABA','left')
    right_GABA_affine_folder = os.path.join('data','skeletons','skeletons_affine+rigid','GABA','right')
    left_GABA_diffeo_folder = os.path.join('data','skeletons','skeletons_diffeomorphic','GABA','left')
    right_GABA_diffeo_folder = os.path.join('data','skeletons','skeletons_diffeomorphic','GABA','right')

    left_GABA_affine_pairwise_means = compute_pairwise_distances(left_GABA_affine_folder)
    right_GABA_affine_pairwise_means = compute_pairwise_distances(right_GABA_affine_folder)
    left_GABA_diffeo_pairwise_means = compute_pairwise_distances(left_GABA_diffeo_folder)
    right_GABA_diffeo_pairwise_means = compute_pairwise_distances(right_GABA_diffeo_folder)

    df_GABA_affine = pd.DataFrame({'pairwise_mean': left_GABA_affine_pairwise_means + right_GABA_affine_pairwise_means, 'set': 'Combined'})
    df_GABA_diffeo = pd.DataFrame({'pairwise_mean': left_GABA_diffeo_pairwise_means + right_GABA_diffeo_pairwise_means, 'set': 'Combined'})
    df_GABA_affine['Registration'] = 'affine+rigid'
    df_GABA_diffeo['Registration'] = 'diffeomorphic'
    df_GABA = pd.concat([df_GABA_affine, df_GABA_diffeo], ignore_index=True)

    plot_group(df_GABA, 'figures/Fig3E_GABAskeletondistance.png', 'figures/Fig3E_GABAskeletondistance.eps', {'affine+rigid':'lime','diffeomorphic':'lime'}, 35)

    # inotocin
    left_inotocin_affine_folder = os.path.join('data','skeletons','skeletons_affine+rigid','inotocin','left')
    right_inotocin_affine_folder = os.path.join('data','skeletons','skeletons_affine+rigid','inotocin','right')
    left_inotocin_diffeo_folder = os.path.join('data','skeletons','skeletons_diffeomorphic','inotocin','left')
    right_inotocin_diffeo_folder = os.path.join('data','skeletons','skeletons_diffeomorphic','inotocin','right')

    left_inotocin_affine_pairwise_means = compute_pairwise_distances(left_inotocin_affine_folder)
    right_inotocin_affine_pairwise_means = compute_pairwise_distances(right_inotocin_affine_folder)
    left_inotocin_diffeo_pairwise_means = compute_pairwise_distances(left_inotocin_diffeo_folder)
    right_inotocin_diffeo_pairwise_means = compute_pairwise_distances(right_inotocin_diffeo_folder)

    df_inotocin_affine = pd.DataFrame({'pairwise_mean': left_inotocin_affine_pairwise_means + right_inotocin_affine_pairwise_means, 'set':'Combined'})
    df_inotocin_diffeo = pd.DataFrame({'pairwise_mean': left_inotocin_diffeo_pairwise_means + right_inotocin_diffeo_pairwise_means, 'set':'Combined'})
    df_inotocin_affine['Registration'] = 'affine+rigid'
    df_inotocin_diffeo['Registration'] = 'diffeomorphic'
    df_inotocin = pd.concat([df_inotocin_affine, df_inotocin_diffeo], ignore_index=True)

    plot_group(df_inotocin, 'figures/Fig3F_inotocinskeletondistance.png', 'figures/Fig3F_inotocinskeletondistance.eps', {'affine+rigid':'cyan','diffeomorphic':'cyan'}, 120)

    print('Done')
