"""Headless runner for MB_Tilt_102725 notebook actions.
Saves two figures to figures/: histogram and polar histogram.
"""
import os
import sys
os.environ.setdefault('MPLBACKEND', 'Agg')
# Ensure pyvista/VTK run off-screen when this script is executed
os.environ.setdefault('PYVISTA_OFF_SCREEN', '1')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import skew, kurtosis
import subprocess
DATA_CSV = 'data/MB_Tilt_Data_102725.csv'
OUT_DIR = 'figures'
os.makedirs(OUT_DIR, exist_ok=True)

# Prefer executing the canonical notebook if available. This keeps the
# single-source-of-truth for plotting in the notebook while allowing CI to
# run the same notebook via this headless script.
NOTEBOOK = 'FrankandLopes_refbrain_code_fig1.ipynb'
executed_nb = os.path.splitext(NOTEBOOK)[0] + '_executed.ipynb'
def try_execute_notebook():
    """Attempt to execute the canonical notebook via nbconvert. Returns True on success."""
    if not os.path.exists(NOTEBOOK):
        print(f'Notebook not found: {NOTEBOOK} — falling back to inline CSV logic')
        return False
    # Build the nbconvert command using the current Python executable
    cmd = [sys.executable, '-m', 'nbconvert', '--to', 'notebook', '--execute', NOTEBOOK,
           '--ExecutePreprocessor.timeout=600', '--output', executed_nb]
    print('Running nbconvert to execute notebook:', ' '.join(cmd))
    try:
        completed = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=os.environ)
        print('nbconvert stdout:\n', completed.stdout.decode('utf-8'))
        print('nbconvert stderr:\n', completed.stderr.decode('utf-8'))
        print('Notebook executed successfully — check figures in', OUT_DIR)
        return True
    except subprocess.CalledProcessError as e:
        print('nbconvert failed with return code', e.returncode)
        try:
            print('stdout:\n', e.stdout.decode('utf-8'))
            print('stderr:\n', e.stderr.decode('utf-8'))
        except Exception:
            pass
        print('Falling back to inline CSV plotting logic')
        return False

# Try to run the notebook first; if it fails, run the inline CSV-based plotting.
if try_execute_notebook():
    sys.exit(0)

# Only accept CSV now
if os.path.exists(DATA_CSV):
    df = pd.read_csv(DATA_CSV)
    print(f"Loaded {DATA_CSV}")
else:
    print(f"ERROR: data file not found: {DATA_CSV}")
    sys.exit(2)
# Use 6th column (index 5) as in notebook
column = df.iloc[:, 5]
values = column.dropna().values

# Histogram
bin_edges = np.linspace(-50, 50, 11)
plt.figure()
plt.hist(values, bins=bin_edges, edgecolor='k')
plt.title('MB Tilt Quantification')
plt.xlabel('Tilt Angle')
plt.ylabel('Frequency')
plt.xlim(-50, 50)
hist_png = os.path.join(OUT_DIR, 'MB_Tilt_hist.png')
hist_eps = os.path.join(OUT_DIR, 'MB_Tilt_hist.eps')
plt.savefig(hist_png, dpi=150)
try:
    plt.savefig(hist_eps)
except Exception:
    pass
plt.close()
print(f"WROTE: {hist_png}, {hist_eps}")

# Polar histogram
angles_rad = np.deg2rad(values)
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, polar=True)
ax.hist(angles_rad, bins=10, edgecolor='black')
ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)
plt.title('Polar Histogram of MB Tilts')
polar_png = os.path.join(OUT_DIR, 'MB_Tilt_polar.png')
polar_eps = os.path.join(OUT_DIR, 'MB_Tilt_polar.eps')
plt.savefig(polar_png, dpi=150)
try:
    plt.savefig(polar_eps)
except Exception:
    pass
plt.close()
print(f"WROTE: {polar_png}, {polar_eps}")
print('Done')
