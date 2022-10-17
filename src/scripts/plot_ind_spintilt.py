#!/usr/bin/env python

import paths
import numpy as np
import matplotlib.pyplot as plt
from utils import plot_mean_and_90CI, plot_o3b_spintilt, load_ind_tilt_ppd

figx, figy = 7,5
fig, ax = plt.subplots(nrows=1, ncols=1, sharey='row', sharex='row', figsize=(figx,figy))
lab='cos_tilt_1'
ax = plot_o3b_spintilt(ax,'o1o2o3_mass_c_iid_mag_iid_tilt_powerlaw_redshift_orientation_data.h5', ct1=lab=='cos_tilt_1', lab='Abbott et. al. 2021b', col='tab:blue')

xmin, xmax = -1, 1
xs, ct1_pdfs, ct2_pdfs = load_ind_tilt_ppd()
for i in range(len(ct1_pdfs)):
    ct1_pdfs[i] /= np.trapz(ct1_pdfs[i], xs)
    ct2_pdfs[i] /= np.trapz(ct2_pdfs[i], xs)
xmin=-1
maxy=0
for ps, lab, c in zip([ct1_pdfs, ct2_pdfs], ['cos_tilt_1', 'cos_tilt_2'],  ['tab:orange', 'tab:olive']):
    if lab == 'cos_tilt_1':
        label=r'This Work ($p(\cos{\theta_1})$)'
    else:
        label=r'This Work ($p(\cos{\theta_2})$)'
    ax = plot_mean_and_90CI(ax, xs, ps, color=c,label=label)
    high = np.percentile(ps, 95, axis=0)
    if max(high) > maxy:
        maxy = max(high)

ax.set_xlabel(r'$\cos{\theta}$', fontsize=18)
ax.set_ylabel(r'$p(\cos{\theta})$', fontsize=18)

ax.set_xlim(xmin, xmax)

ax.set_ylim(0, 1.6)
ax.legend(frameon=False, fontsize=14, loc='upper left');
ax.grid(True, which="major", ls=":")
ax.tick_params(labelsize=14)
    
plt.title(f'GWTC-3: Spin Tilt Distribution', fontsize=18);
fig.tight_layout()
plt.savefig(paths.figures / 'ind_spintilt.pdf', dpi=300);