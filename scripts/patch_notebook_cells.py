"""Patch the Fig3 notebook by replacing unsafe medians[reg] indexing and plt.show() calls
with safe lookups and headless-friendly saves. Writes back the notebook file.
"""
import nbformat
from pathlib import Path

nb_path = Path('FrankandLopes_refbrain_code_fig3.ipynb')
nb = nbformat.read(nb_path, as_version=4)

replaced = 0
for cell in nb.cells:
    if cell.cell_type != 'code':
        continue
    src = ''.join(cell.source) if isinstance(cell.source, list) else str(cell.source)
    if 'medians[reg]' in src or "plt.show()" in src or 'Fig3E_GABAskeletondistance.png' in src:
        # do safe replacements
        src = src.replace("plt.hlines(medians[reg], xmin=i - 0.2, xmax=i + 0.2, colors='black', linewidth=5)",
                          "med_val = medians.get(reg, np.nan) if hasattr(medians,'get') else (medians[reg] if reg in medians.index else np.nan)\n    std_val = stds.get(reg, np.nan) if hasattr(stds,'get') else (stds[reg] if reg in stds.index else np.nan)\n    if not np.isnan(med_val):\n        plt.hlines(med_val, xmin=i - 0.2, xmax=i + 0.2, colors='black', linewidth=5)")
        src = src.replace("plt.errorbar(x=i, y=medians[reg], yerr=stds[reg], fmt='none',", "# errorbar replaced with safe lookup\n    if not np.isnan(std_val):\n        plt.errorbar(x=i, y=med_val, yerr=std_val, fmt='none',")
        src = src.replace("plt.show()", "# headless: plt.show() removed")
        cell.source = src
        replaced += 1

nbformat.write(nb, nb_path)
print('patched cells:', replaced)
