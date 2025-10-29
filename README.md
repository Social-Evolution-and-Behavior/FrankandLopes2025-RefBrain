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

## Quick notes

- Prefer conda-forge for binary packages (VTK/pyvista): these are provided as pre-built binaries on conda-forge and avoid long/fragile pip builds. Example: `conda install -c conda-forge pyvista vtk`.
 
