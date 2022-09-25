#!/usr/bin/env python

import paths
import numpy as np
import matplotlib.pyplot as plt
import deepdish as dd
from utils import plot_mean_and_90CI


def load_tilt_ppd():
    datadict = dd.io.load(paths.data / 'mspline_50m1_16ind_compspins_smoothprior_powerlaw_q_z_ppds.h5')
    return datadict['dRdct1'], datadict['dRdct2'], datadict['tilts'], datadict['tilts']


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

figx, figy = 7,5
fig, ax = plt.subplots(nrows=1, ncols=1, sharey='row', sharex='row', figsize=(figx,figy))
lab='cos_tilt_1'
ax = plot_o3b_spintilt(ax,'o1o2o3_mass_c_iid_mag_iid_tilt_powerlaw_redshift_orientation_data.h5', ct1=lab=='cos_tilt_1', lab='Default', col='tab:blue')

xmin, xmax = -1, 1
ct1_pdfs, ct2_pdfs, xs, xs = load_tilt_ppd()
for i in range(len(ct1_pdfs)):
    ct1_pdfs[i] /= np.trapz(ct1_pdfs[i], xs)
    ct2_pdfs[i] /= np.trapz(ct2_pdfs[i], xs)
xmin=-1
maxy=0
for ps, lab, c in zip([ct1_pdfs, ct2_pdfs], ['cos_tilt_1', 'cos_tilt_2'],  ['tab:red', 'tab:olive']):
    ax = plot_mean_and_90CI(ax, xs, ps, color=c, label=f'MSpline-{lab}')
    high = np.percentile(ps, 95, axis=0)
    if max(high) > maxy:
        maxy = max(high)

ax.set_xlabel(r'$\cos{\theta}$', fontsize=18)
ax.set_ylabel(r'$p(\cos{\theta})$', fontsize=18)

ax.set_xlim(xmin, xmax)

ax.set_ylim(0, maxy)
ax.legend(frameon=False, fontsize=14);
ax.grid(True, which="major", ls=":")
ax.tick_params(labelsize=14)
    
plt.title(f'GWTC-3: MSpline Spin Tilt Distribution', fontsize=18);
fig.tight_layout()
plt.savefig(paths.figures / 'ind_spintilt.pdf', dpi=300);