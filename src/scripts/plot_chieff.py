import paths
import numpy as np
import matplotlib.pyplot as plt
import deepdish as dd
import json
from utils import plot_mean_and_90CI

fig, ax = plt.subplots(1,1,figsize=(7,5))
ppds = dd.io.load(paths.data / "chi_eff_ppds.h5")
default = ppds["Default"]
mspl = ppds["BSplineInd"]
mspl2 = ppds['BSplineIID']
ax = plot_mean_and_90CI(ax, default["chieffs"], default["pchieff"], color='tab:blue', label='Default \n(Abbott et. al. 2021b)', bounds=False)

with open(paths.data / "gaussian-spin-xeff-xp-ppd-data.json", 'r') as jf:
    o3b_gaussian_data = json.load(jf)
    o3b_gaussian_chi_eff_grid = np.array(o3b_gaussian_data['chi_eff_grid'])
    o3b_gaussian_chi_eff_data = np.array(o3b_gaussian_data['chi_eff_pdfs'])

ax = plot_mean_and_90CI(ax, o3b_gaussian_chi_eff_grid, o3b_gaussian_chi_eff_data, color='k', label='Gaussian \n(Abbott et. al. 2021b)', bounds=False)
ax = plot_mean_and_90CI(ax, mspl["chieffs"], mspl["pchieff"], color='tab:orange', label='This Work (Ind Spin)', bounds=False)

#handpicked_ppds = dd.io.load(
#    paths.data / "BSpline_50m1_24chieff_smoothprior_powerlaw_q_z_fitlamb_ppds.h5"
#)
#xs = handpicked_ppds["chieffs"]
#pchiefss = handpicked_ppds["dRdchieff"]
#for i in range(1500):
#    norm = np.trapz(pchiefss[i, :], x=xs)
#    pchiefss[i, :] /= norm
#ax = plot_mean_and_90CI(ax, xs, pchiefss, color='tab:red', label='BSpline Chi Eff')

ax = plot_mean_and_90CI(ax, mspl2["chieffs"], mspl2["pchieff"], color='tab:red', label='This Work (IID Spin)', bounds=True, fill_alpha=0.125)
plt.xlim(-0.65, 0.5)
plt.ylim(0, 5.25)
plt.legend(frameon=False, fontsize=14, loc='upper left')
ax.grid(True, which="major", ls=":")
ax.tick_params(labelsize=14)
plt.xlabel(r"$\chi_\mathrm{eff}$", fontsize=18)
plt.ylabel(r"$p(\chi_\mathrm{eff})$", fontsize=18)
plt.title(r"GWTC-3: $\chi_\mathrm{eff}$ Distribution", fontsize=18)
fig.tight_layout()
plt.savefig(paths.figures / "chi_eff.pdf", dpi=300)