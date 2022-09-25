#!/usr/bin/env python

import paths
import numpy as np
import matplotlib.pyplot as plt
import deepdish as dd
from utils import plot_mean_and_90CI

def load_mag_ppd():
    datadict = dd.io.load(paths.data / 'mspline_50m1_16iid_compspins_smoothprior_powerlaw_q_z_ppds.h5')
    return datadict['mags'], datadict['dRda']

def plot_o3b_spinmag(ax, fi, a1=True, col='tab:blue', lab='PP'):
    xs = np.linspace(0, 1, 1000)
    _data = dd.io.load(paths.data / fi)
    lines = dict()
    for key in _data["lines"].keys():
        lines[key] = _data["lines"][key][()]
        for ii in range(len(lines[key])):
            lines[key][ii] /= np.trapz(lines[key][ii], xs)
    if a1:
        ax = plot_mean_and_90CI(ax, xs, lines['a_1'], color=col, label=lab, bounds=False)
    else:
        ax = plot_mean_and_90CI(ax, xs, lines['a_2'], color=col, label=lab, bounds=False)
    return ax

figx, figy = 7, 5
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(figx,figy))

xmin, xmax = 0, 1
mags, a_pdfs = load_mag_ppd()
for jj in range(len(a_pdfs)):
    a_pdfs[jj,:] /= np.trapz(a_pdfs[jj,:], mags)
ax = axs
ax = plot_o3b_spinmag(ax,'o1o2o3_mass_c_iid_mag_iid_tilt_powerlaw_redshift_magnitude_data.h5', lab='Default', col='tab:blue')
ax = plot_mean_and_90CI(ax, mags, a_pdfs, color='tab:red', label='MSpline')
high = np.percentile(a_pdfs, 95, axis=0)
ax.legend(frameon=False, fontsize=14);
ax.set_xlabel(r'$a$', fontsize=18)
ax.set_ylabel(r'$p(a)$', fontsize=18)

ax.set_xlim(xmin, xmax)
ax.set_ylim(0, max(high))
ax.legend(frameon=False, fontsize=14);
ax.grid(True, which="major", ls=":")
ax.tick_params(labelsize=14)

plt.suptitle(f'GWTC-3: MSpline Spin Mag Distribution', fontsize=18);
fig.tight_layout()
plt.savefig(paths.figures / 'iid_spinmag.pdf', dpi=300);