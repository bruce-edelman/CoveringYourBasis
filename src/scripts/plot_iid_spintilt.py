#!/usr/bin/env python

import paths
import numpy as np
import matplotlib.pyplot as plt
import deepdish as dd
from utils import plot_mean_and_90CI

def load_tilt_ppd():
    datadict = dd.io.load(paths.data / 'mspline_50m1_16iid_compspins_smoothprior_powerlaw_q_z_ppds.h5')
    return datadict['tilts'], datadict['dRdct']

def plot_o3b_spintilt(ax, fi,ct1=False, col='tab:blue', lab='PP'):
    xs = np.linspace(-1, 1, 1000)
    _data = dd.io.load(paths.data / fi)
    lines = dict()
    for key in _data["lines"].keys():
        lines[key] = _data["lines"][key][()]
        for ii in range(len(lines[key])):
            lines[key][ii] /= np.trapz(lines[key][ii], xs)
    if ct1:
        ax = plot_mean_and_90CI(ax, xs, lines['cos_tilt_1'], color=col, label=lab, bounds=False)
    else:
        ax = plot_mean_and_90CI(ax, xs, lines['cos_tilt_2'], color=col, label=lab, bounds=False)
    return ax

figx, figy = 7, 5
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(figx,figy))
xmin, xmax = -1, 1
xs, ct_pdfs = load_tilt_ppd()
for jj in range(len(ct_pdfs)):
    ct_pdfs[jj,:] /= np.trapz(ct_pdfs[jj,:], xs)

xmin=-1
ax = axs
ax = plot_o3b_spintilt(ax,'o1o2o3_mass_c_iid_mag_iid_tilt_powerlaw_redshift_orientation_data.h5', ct1=True, lab='Default', col='tab:blue')
ax = plot_mean_and_90CI(ax, xs, ct_pdfs, color='tab:red', label='MSpline')
high = np.percentile(ct_pdfs, 95, axis=0)
ax.set_xlabel(r'$\cos{\theta}$', fontsize=18)
ax.set_ylabel(r'$p(\cos{\theta})$', fontsize=18)

ax.set_xlim(xmin, xmax)
ax.legend(frameon=False, fontsize=14);
ax.grid(True, which="major", ls=":")
ax.tick_params(labelsize=14)
ax.set_ylim(0, max(high))

plt.suptitle(f'GWTC-3: MSpline Spin Tilt Distribution', fontsize=18);
fig.tight_layout()
plt.savefig(paths.figures / 'iid_spintilt.pdf', dpi=300);