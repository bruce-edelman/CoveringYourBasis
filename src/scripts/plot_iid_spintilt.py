#!/usr/bin/env python

import paths
import numpy as np
import matplotlib.pyplot as plt
from utils import plot_mean_and_90CI, load_iid_tilt_ppd, plot_o3b_spintilt

figx, figy = 7, 5
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(figx,figy))
xmin, xmax = -1, 1
xs, ct_pdfs = load_iid_tilt_ppd()
for jj in range(len(ct_pdfs)):
    ct_pdfs[jj,:] /= np.trapz(ct_pdfs[jj,:], xs)

xmin=-1
ax = axs
ax = plot_o3b_spintilt(ax,'o1o2o3_mass_c_iid_mag_iid_tilt_powerlaw_redshift_orientation_data.h5', ct1=True, lab='Abbott et. al. 2021b', col='tab:blue')
ax = plot_mean_and_90CI(ax, xs, ct_pdfs, color='tab:red', label='This Work')
high = np.percentile(ct_pdfs, 95, axis=0)
ax.set_xlabel(r'$\cos{\theta}$', fontsize=18)
ax.set_ylabel(r'$p(\cos{\theta})$', fontsize=18)

ax.set_xlim(xmin, xmax)
ax.legend(frameon=False, fontsize=14, loc='upper left');
ax.grid(True, which="major", ls=":")
ax.tick_params(labelsize=14)
ax.set_ylim(0, 1.6)

plt.title(f'GWTC-3: Spin Tilt Distribution', fontsize=18);
fig.tight_layout()
plt.savefig(paths.figures / 'iid_spintilt.pdf', dpi=300);