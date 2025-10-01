#!/usr/bin/env python3
"""Headless runner to compute Fig2 and Supp Fig2 plots.
Produces the same PNG/EPS files in the figures/ directory.

Usage: run from the repository root. Requires data/brainvolume/refbrain_brainvolumedata.csv
and the mesh tifs under data/brainvolume/brain_meshes/. PyVista must be available for mesh rendering.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress, levene, mannwhitneyu
from skimage import measure
import tifffile
import pyvista as pv


def safe_mkdir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def safe_read_csv(path):
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        print(f"ERROR: failed to read CSV {path}: {e}")
        return pd.DataFrame()


def plot_simple_strip(df, y_col, out_png, out_eps, xlabel=None, ylabel=None, ylim=None, xticks=None, figsize=(4,6)):
    if df.empty:
        print(f"No data for {out_png} — skipping")
        return
    plt.figure(figsize=figsize)
    sns.stripplot(x=[0] * len(df), y=df[y_col], color='black', jitter=0.15, size=10, alpha=0.6)
    if ylim is not None:
        plt.ylim(*ylim)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    # median line
    med = df[y_col].median()
    plt.hlines(y=med, xmin=-0.2, xmax=0.2, colors='magenta', linewidth=2)
    if xticks is not None:
        plt.xticks(xticks)
    sns.despine(top=True, right=True)
    plt.tight_layout()
    safe_mkdir(out_png)
    plt.savefig(out_eps, format='eps', dpi=300)
    plt.savefig(out_png, format='png', dpi=600)
    plt.close()
    print(f'WROTE: {out_png}, {out_eps}')


def plot_regression(df, x_col, y_col, out_png, out_eps, xlim=None, xticks=None, ylim=None, xlabel=None, ylabel=None, figsize=(6,6)):
    df_clean = df.dropna(subset=[x_col, y_col])
    if df_clean.empty:
        print(f"No data for regression {out_png} — skipping")
        return
    plt.figure(figsize=figsize)
    plt.scatter(df_clean[x_col], df_clean[y_col], alpha=1, color='black', s=80)
    try:
        slope, intercept, r_value, p_value, std_err = linregress(df_clean[x_col], df_clean[y_col])
        x_vals = np.linspace(df_clean[x_col].min(), df_clean[x_col].max(), 100)
        y_vals = slope * x_vals + intercept
        plt.plot(x_vals, y_vals, color='magenta', linewidth=2, linestyle='--')
        n = len(df_clean)
        r_squared = r_value ** 2
        f_statistic = (r_squared / (1 - r_squared)) * (n - 2) if (1 - r_squared) != 0 else np.nan
        regression_text = (
            f"Slope: {slope:.2e}\n"
            f"Intercept: {intercept:.2e}\n"
            f"R²: {r_value**2:.3f}\n"
            f"P-value: {p_value:.2e}\n"
            f"Std. Error: {std_err:.2e}\n"
            f"F-statistic: {f_statistic:.2f}\n"
        )
        plt.text(0.02, 0.98, regression_text, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', horizontalalignment='left', bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))
    except Exception as e:
        print(f"WARNING: regression failed for {out_png}: {e}")
    if xlim is not None:
        plt.xlim(*xlim)
    if xticks is not None:
        plt.xticks(xticks)
    if ylim is not None:
        plt.ylim(*ylim)
    if xlabel:
        plt.xlabel(xlabel, fontsize=14)
    if ylabel:
        plt.ylabel(ylabel, fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    sns.despine(top=True, right=True)
    plt.tight_layout()
    safe_mkdir(out_png)
    plt.savefig(out_eps, format='eps', dpi=300)
    plt.savefig(out_png, format='png', dpi=600)
    plt.close()
    print(f'WROTE: {out_png}, {out_eps}')


def render_mesh(tif_path, out_png, color='cyan', scale_bar_length_microns=50, voxel_size=0.8, smooth_iters=300, decimate_fraction=0.2, window_size=(1920,1080)):
    if not os.path.exists(tif_path):
        print(f"Mesh tif not found: {tif_path} — skipping render")
        return
    try:
        img = tifffile.imread(tif_path)
        # assume integer labels and target label 1
        mask = (img == 1)
        if mask.sum() == 0:
            print(f"No label 1 found in {tif_path} — skipping")
            return
        verts, faces, normals, values = measure.marching_cubes(mask, level=0)
        faces_flat = np.hstack([[3] + list(face) for face in faces])
        mesh = pv.PolyData(verts, faces_flat)
        simplified = mesh.decimate_pro(decimate_fraction) if decimate_fraction and decimate_fraction < 1.0 else mesh
        smoothed = simplified.smooth(n_iter=smooth_iters)
        pv.OFF_SCREEN = True
        plotter = pv.Plotter(off_screen=True)
        plotter.add_mesh(smoothed, color=color, smooth_shading=True)
        plotter.set_background('white')
        bounds = smoothed.bounds
        start = [bounds[0], bounds[2], bounds[4]]
        end = [bounds[0], bounds[2], bounds[4] + (scale_bar_length_microns/voxel_size)]
        scale_bar = pv.Line(start, end)
        plotter.add_mesh(scale_bar, color='black', line_width=5)
        plotter.view_vector((-1,0,0), viewup=(0,-1,0))
        safe_mkdir(out_png)
        plotter.screenshot(out_png, window_size=window_size)
        plotter.close()
        print(f'WROTE: {out_png}')
    except Exception as e:
        print(f"ERROR rendering mesh {tif_path}: {e}")


def main():
    csv_path = os.path.join('data','brainvolume','refbrain_brainvolumedata.csv')
    df = safe_read_csv(csv_path)
    if df.empty:
        print('No brainvolume CSV found or file empty — exiting')
        return

    # split experiments
    refbrain_df = df[df['Experiment']=='refbrain']
    lineA_df = df[df['Experiment']=='lineA']
    intercastevworker_df = df[df['Experiment']=='intercastevworker']
    brainvbody_df = df[df['Experiment']=='brainvbody']

    # Fig2A
    filtered = refbrain_df[refbrain_df['Condition']=='included']
    plot_simple_strip(filtered, 'AMIRA volume', 'figures/Fig2A_indvbrainvolumesinrefbrain.png', 'figures/Fig2A_indvbrainvolumesinrefbrain.eps', xlabel='40 Brains in\nReference Dataset', ylabel='Total Brain Volume (μm³)', ylim=(0, 1.3e7))

    # Fig2B - render largest and smallest brains
    mesh_path = os.path.join('data','brainvolume','brain_meshes')
    largest_brain = 'synA647_LL_4_220214_resampled_0.8x0.8x0.processed0_manualfix.tif'
    smallest_brain = 'synA647_LL_201213_263_resampled_0.8x0.8x0.processed0_manualfix.tif'
    render_mesh(os.path.join(mesh_path, largest_brain), 'figures/Fig2B_largestbrain_withscale.png', color='cyan')
    render_mesh(os.path.join(mesh_path, smallest_brain), 'figures/Fig2B_smallestbrain_withscale.png', color='orange')

    # Fig2C-F: regressions
    plot_regression(refbrain_df, 'Avg AL' if 'Avg AL' in refbrain_df.columns else 'R AL', 'AMIRA volume', 'figures/Fig2C_brainvolumevsALvolume.png', 'figures/Fig2C_brainvolumevsALvolume.eps', xlim=(1.6e5,6.0e5), xticks=np.arange(2e5,7e5,1e5), ylim=(0,1.3e7), xlabel='Average AL Volume (μm³)', ylabel='Total Brain Volume (μm³)')
    # For MB, OL, CX the notebook computes avg MB/OL etc — recompute defensively
    if 'R MB' in refbrain_df.columns and 'L MB' in refbrain_df.columns:
        refbrain_df['avg MB'] = (refbrain_df.get('R MB', np.nan) + refbrain_df.get('L MB', np.nan)) / 2
        plot_regression(refbrain_df, 'avg MB', 'AMIRA volume', 'figures/Fig2D_brainvolumevsMBvolume.png', 'figures/Fig2D_brainvolumevsMBvolume.eps', xlim=(3.5e5,9.5e5), xticks=np.arange(4e5,9.1e5,1e5), ylim=(0,1.3e7), xlabel='Average MB Volume (μm³)', ylabel='Total Brain Volume (μm³)')
    if 'R OL' in refbrain_df.columns and 'L OL' in refbrain_df.columns:
        refbrain_df['avg OL'] = (refbrain_df.get('R OL', np.nan) + refbrain_df.get('L OL', np.nan)) / 2
        plot_regression(refbrain_df, 'avg OL', 'AMIRA volume', 'figures/Fig2E_brainvolumevsOLvolume.png', 'figures/Fig2E_brainvolumevsOLvolume.eps', xlim=(4e3,19e3), xticks=np.arange(5e3,19e3,3e3), ylim=(0,1.3e7), xlabel='Average OL Volume (μm³)', ylabel='Total Brain Volume (μm³)')
    if 'CX' in refbrain_df.columns:
        plot_regression(refbrain_df, 'CX', 'AMIRA volume', 'figures/Fig2F_brainvolumevsCXvolume.png', 'figures/Fig2F_brainvolumevsCXvolume.eps', xlim=(2e4,7.1e4), xticks=np.arange(2e4,7e4,1e4), ylim=(0,1.3e7), xlabel='CX Volume (μm³)', ylabel='Total Brain Volume (μm³)')

    # Supp Fig 2A - egocentric leaning comparison
    filtered = refbrain_df[refbrain_df['Condition']=='included']
    if not filtered.empty and 'Egocentric Leaning' in filtered.columns:
        left_volumes = filtered[filtered['Egocentric Leaning']=='left']['AMIRA volume']
        right_volumes = filtered[filtered['Egocentric Leaning']=='right']['AMIRA volume']
        try:
            bf_stat, bf_p = levene(left_volumes, right_volumes, center='median')
            lev_stat, lev_p = levene(left_volumes, right_volumes, center='mean')
            mw_stat, mw_p = mannwhitneyu(left_volumes, right_volumes, alternative='two-sided')
        except Exception as e:
            bf_p = lev_p = mw_p = np.nan
        plt.figure(figsize=(6,6))
        sns.stripplot(x='Egocentric Leaning', y='AMIRA volume', data=filtered, order=['left','right'], jitter=0.15, size=10, alpha=0.6, color='black')
        plt.ylim(0,1.3e7)
        plt.xlim(-0.5,1.5)
        medians = filtered.groupby('Egocentric Leaning')['AMIRA volume'].median()
        # draw median lines
        if 'left' in medians.index and 'right' in medians.index:
            plt.hlines(y=medians.values, xmin=[-0.2,0.8], xmax=[0.2,1.2], colors='magenta', linewidth=2)
        plt.title(f"Variance Tests:\nBrown-Forsythe (median): p = {bf_p:.3g} | Levene (mean): p = {lev_p:.3g}", fontsize=11)
        sns.despine(top=True, right=True)
        plt.tight_layout()
        safe_mkdir('figures/SF2A_egocentricleaning_vs_volume.png')
        plt.savefig('figures/SF2A_egocentricleaning_vs_volume.eps', format='eps', dpi=300)
        plt.savefig('figures/SF2A_egocentricleaning_vs_volume.png', format='png', dpi=600)
        plt.close()
        print('WROTE: figures/SF2A_egocentricleaning_vs_volume.png, .eps')

    # Supp Fig 2B - lineA
    plot_simple_strip(lineA_df, 'AMIRA volume', 'figures/SF2B_lineA_volume.png', 'figures/SF2B_lineA_volume.eps', xlabel='Line A', ylabel='Total Brain Volume (μm³)', ylim=(0,1.3e7), figsize=(4,6))

    # Supp Fig 2C - worker vs intercaste
    if not intercastevworker_df.empty:
        plt.figure(figsize=(6,6))
        sns.stripplot(x='Condition', y='AMIRA volume', data=intercastevworker_df, order=['worker','intercaste'], jitter=0.15, size=10, alpha=0.6, color='black')
        plt.ylim(0,1.3e7)
        plt.xlim(-0.5,1.5)
        plt.xticks(fontsize=14); plt.yticks(fontsize=14)
        medians = intercastevworker_df.groupby('Condition')['AMIRA volume'].median()
        try:
            plt.hlines(y=[medians.get('worker', np.nan), medians.get('intercaste', np.nan)], xmin=[-0.2,0.8], xmax=[0.2,1.2], colors='magenta', linewidth=2)
        except Exception:
            pass
        plt.ylabel('Total Brain Volume (μm³)')
        plt.xlabel('')
        plt.title(f"Variance Tests:\nBrown-Forsythe (median): p = {np.nan} | Levene (mean): p = {np.nan}", fontsize=11)
        sns.despine(top=True, right=True)
        plt.tight_layout()
        safe_mkdir('figures/SF2C_worker_vs_intercaste.png')
        plt.savefig('figures/SF2C_worker_vs_intercaste.eps', format='eps', dpi=300)
        plt.savefig('figures/SF2C_worker_vs_intercaste.png', format='png', dpi=600)
        plt.close()
        print('WROTE: figures/SF2C_worker_vs_intercaste.png, .eps')

    # Supp Fig 2E and 2F - regressions vs body size / head area
    plot_regression(brainvbody_df, 'body length', 'AMIRA volume', 'figures/SF2E_brainvbodylength.png', 'figures/SF2E_brainvbodylength.eps', xlim=(1.45,1.71), xticks=[1.5,1.6,1.7], ylim=(0,1.3e7), xlabel='Body Size (mm)', ylabel='Total Brain Volume (μm³)', figsize=(4,6))
    plot_regression(brainvbody_df, 'head area', 'AMIRA volume', 'figures/SF2F_brainvheadarea.png', 'figures/SF2F_brainvheadarea.eps', xlim=(0.165,0.22), xticks=[0.17,0.19,0.21], ylim=(0,1.3e7), xlabel='Head Area (mm²)', ylabel='Total Brain Volume (μm³)', figsize=(4,6))

    print('Done')


if __name__ == '__main__':
    main()
