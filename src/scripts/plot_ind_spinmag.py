#!/usr/bin/env python

import paths
import numpy as np
import matplotlib.pyplot as plt
import deepdish as dd

def load_mag_ppd():
    datadict = dd.io.load(paths.data / 'mspline_50m1_16ind_compspins_smoothprior_powerlaw_q_z_ppds.h5')
    return datadict['dRda1'], datadict['dRda2'], datadict['mags'], datadict['mags']

def plot_o3b_spinmag(ax, fi, a1=False, col='tab:blue', lab='PP'):
    xs = np.linspace(0, 1, 1000)
    
    _data = dd.io.load(paths.data / fi)
    lines = dict()
    for key in _data["lines"].keys():
        lines[key] = _data["lines"][key][()]
        for ii in range(len(lines[key])):
            lines[key][ii] /= np.trapz(lines[key][ii], xs)
    if a1:
        low = np.percentile(lines['a_1'], 35, axis=0)
        high = np.percentile(lines['a_1'], 65, axis=0)
        ax.fill_between(xs, low, high, color=col, alpha=0.5, label=lab)
        ax.plot(xs, low, color='k', lw=0.25, alpha=0.1)#, label=lab)
        ax.plot(xs, high, color='k', lw=0.25, alpha=0.1)#, label=lab)
        low = np.percentile(lines['a_1'], 5, axis=0)
        high = np.percentile(lines['a_1'], 95, axis=0)
        ax.fill_between(xs, low, high, color=col, alpha=0.1)
        ax.plot(xs, low, color='k', lw=0.05, alpha=0.05)#, label=lab)
        ax.plot(xs, high, color='k', lw=0.05, alpha=0.05)#, label=lab)
    else:
        low = np.percentile(lines['a_2'], 35, axis=0)
        high = np.percentile(lines['a_2'], 65, axis=0)
        ax.fill_between(xs, low, high, color=col, alpha=0.5, label=lab)
        ax.plot(xs, low, color='k', lw=0.25, alpha=0.1)#, label=lab)
        ax.plot(xs, high,color='k', lw=0.25, alpha=0.1)
        low = np.percentile(lines['a_2'], 5, axis=0)
        high = np.percentile(lines['a_2'], 95, axis=0)
        ax.fill_between(xs, low, high, color=col, alpha=0.1)
        ax.plot(xs, low, color='k', lw=0.05, alpha=0.05)#, label=lab)
        ax.plot(xs, high,color='k', lw=0.05, alpha=0.05)
    return ax

figx, figy = 7, 5
fig, ax = plt.subplots(nrows=1, ncols=1, sharey='row', sharex='row', figsize=(figx,figy))

xmin, xmax = 0, 1
lab='a1'
ax = plot_o3b_spinmag(ax,'o1o2o3_mass_c_iid_mag_iid_tilt_powerlaw_redshift_magnitude_data.h5', a1=lab=='a1', lab='Default', col='tab:blue')

a1_pdfs, a2_pdfs, a1s, a2s = load_mag_ppd()
maxy=0
for i in range(len(a1_pdfs)):
    a1_pdfs[i] /= np.trapz(a1_pdfs[i], a1s)
    a2_pdfs[i] /= np.trapz(a2_pdfs[i], a2s)

for ps, lab, c in zip([a1_pdfs, a2_pdfs], ['a1', 'a2'], ['tab:red', 'tab:orange']):
    low = np.percentile(ps, 35, axis=0)
    high = np.percentile(ps, 65, axis=0)
    ax.plot(a1s, low, color='k', lw=0.25, alpha=0.1)
    ax.plot(a1s, high, color='k', lw=0.25, alpha=0.1)
    ax.fill_between(a1s, low, high, color=c, alpha=0.5, label='MSpline')
    low = np.percentile(ps, 5, axis=0)
    high = np.percentile(ps, 95, axis=0)
    #for _ in range(1000):
    #    idx = np.random.choice(ps.shape[0])
    #    ax.plot(a1s, ps[idx], color='k', lw=0.03, alpha=0.03)
    ax.plot(a1s, low, color='k', lw=0.05, alpha=0.05)
    ax.plot(a1s, high, color='k', lw=0.05, alpha=0.05)
    ax.fill_between(a1s, low, high, color=c, alpha=0.1)
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
plt.title(f'GWTC-3: MSpline Independent Spin Mag (12 knots)', fontsize=18);
fig.tight_layout()
plt.savefig(paths.figures / 'ind_spinmag.pdf', dpi=300);