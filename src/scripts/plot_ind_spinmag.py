#!/usr/bin/env python

import paths
import numpy as np
import matplotlib.pyplot as plt
from utils import plot_mean_and_90CI, plot_o3b_spinmag, load_ind_mag_ppd

figx, figy = 7, 5
fig, ax = plt.subplots(nrows=1, ncols=1, sharey='row', sharex='row', figsize=(figx,figy))

xmin, xmax = 0, 1
lab='a1'
ax = plot_o3b_spinmag(ax,'o1o2o3_mass_c_iid_mag_iid_tilt_powerlaw_redshift_magnitude_data.h5', a1=lab=='a1', lab='Default', col='tab:blue')

a1_pdfs, a2_pdfs, a1s, a2s = load_ind_mag_ppd()
maxy=0
for i in range(len(a1_pdfs)):
    a1_pdfs[i] /= np.trapz(a1_pdfs[i], a1s)
    a2_pdfs[i] /= np.trapz(a2_pdfs[i], a2s)

for ps, lab, c in zip([a1_pdfs, a2_pdfs], ['a1', 'a2'], ['tab:red', 'tab:olive']):
    ax = plot_mean_and_90CI(ax, a1s, ps, color=c, label=f'MSpline-{lab}')
    high = np.percentile(ps, 95, axis=0)
    if max(high) > maxy:
        maxy = max(high)


ax.legend(frameon=False, fontsize=14);
ax.set_xlabel(r'$a$', fontsize=18)
ax.set_ylabel(r'$p(a)$', fontsize=18)
ax.set_xlim(xmin, xmax)

ax.set_ylim(0, maxy)
ax.legend(frameon=False, fontsize=14);
ax.grid(True, which="major", ls=":")
ax.tick_params(labelsize=14)
plt.title(f'GWTC-3: MSpline Spin Mag Distribution', fontsize=18);
fig.tight_layout()
plt.savefig(paths.figures / 'ind_spinmag.pdf', dpi=300);