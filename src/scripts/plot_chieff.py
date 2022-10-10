import paths
import numpy as np
import matplotlib.pyplot as plt
import deepdish as dd
import json
from utils import plot_mean_and_90CI

fig, ax = plt.subplots(1,1,figsize=(7,5))
ppds = dd.io.load(paths.data / "chi_eff_ppds.h5")
default = ppds["Default"]
mspl = ppds["MSplineInd"]
mspl2 = ppds['MSplineIID']
ax = plot_mean_and_90CI(ax, default["chieffs"], default["pchieff"], color='tab:blue', label='Default', bounds=False)
ax = plot_mean_and_90CI(ax, mspl["chieffs"], mspl["pchieff"], color='tab:purple', label='MSpline-Ind Spin', bounds=False)
ax = plot_mean_and_90CI(ax, mspl2["chieffs"], mspl2["pchieff"], color='tab:red', label='MSpline-IID Spin', bounds=True, fill_alpha=0.125)

#handpicked_ppds = dd.io.load(
#    paths.data / "mspline_50m1_24chieff_smoothprior_powerlaw_q_z_fitlamb_ppds.h5"
#)
#xs = handpicked_ppds["chieffs"]
#pchiefss = handpicked_ppds["dRdchieff"]
#for i in range(1500):
#    norm = np.trapz(pchiefss[i, :], x=xs)
#    pchiefss[i, :] /= norm
#ax = plot_mean_and_90CI(ax, xs, pchiefss, color='tab:red', label='MSpline Chi Eff')

with open(paths.data / "gaussian-spin-xeff-xp-ppd-data.json", 'r') as jf:
    o3b_gaussian_data = json.load(jf)
    o3b_gaussian_chi_eff_grid = np.array(o3b_gaussian_data['chi_eff_grid'])
    o3b_gaussian_chi_eff_data = np.array(o3b_gaussian_data['chi_eff_pdfs'])
ax = plot_mean_and_90CI(ax, o3b_gaussian_chi_eff_grid, o3b_gaussian_chi_eff_data, color='tab:green', label='Gaussian', bounds=False)

plt.xlim(-0.55, 0.5)
plt.ylim(0, 6)
plt.legend(frameon=False, fontsize=14, loc='upper left')
ax.grid(True, which="major", ls=":")
ax.tick_params(labelsize=14)
plt.xlabel(r"$\chi_\mathrm{eff}$", fontsize=18)
plt.ylabel(r"$p(\chi_\mathrm{eff})$", fontsize=18)
plt.title(f"GWTC-3: Effective Spin Distribution", fontsize=18)
fig.tight_layout()
plt.savefig(paths.figures / "chi_eff.pdf", dpi=300)