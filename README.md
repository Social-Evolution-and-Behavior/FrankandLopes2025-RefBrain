# FrankandLopes2025-RefBrain
This repo contains code for analysis and plotting of data in Frank, Lopes, Mohanta, Seckler, Lacroix, and Kronauer 2025.

## Python environment and requirements

This repository contains Jupyter notebooks that use the scientific Python stack and image-processing libraries. Two simple ways to create an environment:

1) Using pip (recommended if you prefer venv/virtualenv):

	Create a virtual environment and install requirements:

	python -m venv .venv
	source .venv/bin/activate
	pip install --upgrade pip
	pip install -r requirements.txt

2) Using conda (recommended for binary image libs):

	conda env create -f environment.yml
	conda activate refbrain-env

If you add new dependencies while working in the repo, please update `requirements.txt` and `environment.yml` accordingly.

## Lockfiles

For exact, reproducible installs the repository includes two lockfiles created from the active development environment:

- `requirements.lock` — pip-style lockfile with exact package==version pins. Use with pip-tools or pip-sync to recreate the same environment:

```bash
# create a fresh venv and install pinned packages
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.lock
```

- `environment.lock.yml` — conda-style lockfile (conda-forge) that pins key conda packages and contains a pip section for the remaining packages. Recreate with:

```bash
conda env create -f environment.lock.yml
conda activate frankandlopes-refbrain-lock
```

Note: The conda lockfile pins major binary packages (e.g., VTK/pyvista) and places pure-python packages under the pip section. If you prefer a different Python version or need to tighten package versions, edit the lockfile accordingly.

## Quick smoke-test

After creating the environment (see above), run a minimal import check to ensure key packages used by the notebooks are available. These commands are intentionally small and safe — they only import libraries and print versions.

Conda (recommended):

```bash
conda create -n refbrain-env python=3.10 -c conda-forge \
	numpy pandas scipy matplotlib seaborn scikit-image tifffile imagecodecs pyvista vtk notebook ipython -y
conda activate refbrain-env
pip install -r requirements.txt  # optional to get any pip-only extras
```

Pip / venv:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Quick import check (runs a tiny Python snippet that prints package versions):

```bash
python - <<'PY'
import sys
import numpy as np
import pandas as pd
import scipy
import matplotlib
import seaborn
import skimage
import tifffile
import imagecodecs
import pyvista
import vtk
print('ok', 'python', sys.version.split()[0], 'numpy', np.__version__, 'pandas', pd.__version__, 'pyvista', pyvista.__version__)
print('VTK version:', vtk.vtkVersion.GetVTKVersion())
PY
```

If the import check runs without errors and prints versions, you have the minimal environment required to run the notebooks interactively. If pyvista/vtk import fails on a headless machine, prefer creating the conda environment on a machine with proper VTK support or install VTK from conda-forge (the command above does this).
