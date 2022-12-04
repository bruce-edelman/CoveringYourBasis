import paths
import numpy as np
import matplotlib.pyplot as plt
import deepdish as dd
import json
from utils import plot_mean_and_90CI

fig, axs = plt.subplots(nrows=1,ncols=2,figsize=(14,5))
ppds = dd.io.load(paths.data / "chi_eff_chi_p_ppds.h5")
default = ppds["Default"]
mspl = ppds["BSplineInd"]
mspl2 = ppds['BSplineIID']

#Chi_Eff
ax = axs[0]
ax = plot_mean_and_90CI(ax, default["chieffs"], default["pchieff"], color='tab:blue', label='Default \n(Abbott et. al. 2021b)', bounds=False)
with open(paths.data / "gaussian-spin-xeff-xp-ppd-data.json", 'r') as jf:
    o3b_gaussian_data = json.load(jf)
    o3b_gaussian_chi_eff_grid = np.array(o3b_gaussian_data['chi_eff_grid'])
    o3b_gaussian_chi_eff_data = np.array(o3b_gaussian_data['chi_eff_pdfs'])
ax = plot_mean_and_90CI(ax, o3b_gaussian_chi_eff_grid, o3b_gaussian_chi_eff_data, color='k', label='Gaussian \n(Abbott et. al. 2021b)', bounds=False)
ax = plot_mean_and_90CI(ax, mspl["chieffs"], mspl["pchieff"], color='tab:orange', label='This Work (Ind Spin)', bounds=False)
ax = plot_mean_and_90CI(ax, mspl2["chieffs"], mspl2["pchieff"], color='tab:red', label='This Work (IID Spin)', bounds=True, fill_alpha=0.125)
ax.set_xlim(-0.65, 0.5)
ax.set_ylim(0, 5.25)
ax.legend(frameon=False, fontsize=14, loc='upper left')
ax.grid(True, which="major", ls=":")
ax.tick_params(labelsize=14)
ax.set_xlabel(r"$\chi_\mathrm{eff}$", fontsize=18)
ax.set_ylabel(r"$p(\chi_\mathrm{eff})$", fontsize=18)

#Chi_P
ax = axs[1]
ax = plot_mean_and_90CI(ax, default["chips"], default["pchip"], color='tab:blue', label='Default \n(Abbott et. al. 2021b)', bounds=False)
with open(paths.data / "gaussian-spin-xeff-xp-ppd-data.json", 'r') as jf:
    o3b_gaussian_data = json.load(jf)
    o3b_gaussian_chi_eff_grid = np.array(o3b_gaussian_data['chi_p_grid'])
    o3b_gaussian_chi_eff_data = np.array(o3b_gaussian_data['chi_p_pdfs'])
ax = plot_mean_and_90CI(ax, o3b_gaussian_chi_eff_grid, o3b_gaussian_chi_eff_data, color='k', label='Gaussian \n(Abbott et. al. 2021b)', bounds=False)
ax = plot_mean_and_90CI(ax, mspl["chips"], mspl["pchip"], color='tab:orange', label='This Work (Ind Spin)', bounds=False)
ax = plot_mean_and_90CI(ax, mspl2["chips"], mspl2["pchip"], color='tab:red', label='This Work (IID Spin)', bounds=True, fill_alpha=0.125)
ax.set_xlim(0, 1.0)
ax.set_ylim(0)
ax.legend(frameon=False, fontsize=14, loc='upper right')
ax.grid(True, which="major", ls=":")
ax.tick_params(labelsize=14)
ax.set_xlabel(r"$\chi_\mathrm{p}$", fontsize=18)
ax.set_ylabel(r"$p(\chi_\mathrm{p})$", fontsize=18)

plt.suptitle(r"GWTC-3: $\chi_\mathrm{eff}$ and $\chi_\mathrm{p}$ Distributions", fontsize=18)
fig.tight_layout()
plt.savefig(paths.figures / "effspin.pdf", dpi=300)