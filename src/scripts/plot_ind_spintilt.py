#!/usr/bin/env python

import paths
import numpy as np
import matplotlib.pyplot as plt
import deepdish as dd

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
        #med = np.median(lines['cos_tilt_1'], axis=0)
        #ax.plot(xs, med, color=col, lw=5, alpha=0.5, label=lab)
        low = np.percentile(lines['cos_tilt_1'], 35, axis=0)
        high = np.percentile(lines['cos_tilt_1'], 65, axis=0)
        ax.fill_between(xs, low, high, color=col, alpha=0.5, label=lab)
        ax.plot(xs, low, color='k', lw=0.25, alpha=0.1)#, label=lab)
        ax.plot(xs, high, color='k', lw=0.25, alpha=0.1)#, label=lab)
        low = np.percentile(lines['cos_tilt_1'], 5, axis=0)
        high = np.percentile(lines['cos_tilt_1'], 95, axis=0)
        ax.fill_between(xs, low, high, color=col, alpha=0.1)
        ax.plot(xs, low, color='k', lw=0.05, alpha=0.05)#, label=lab)
        ax.plot(xs, high, color='k', lw=0.05, alpha=0.05)#, label=lab)
    else:
        #med = np.median(lines['cos_tilt_2'], axis=0)
        #ax.plot(xs, med, color=col, lw=5, alpha=0.5, label=lab)
        low = np.percentile(lines['cos_tilt_2'], 35, axis=0)
        high = np.percentile(lines['cos_tilt_2'], 65, axis=0)
        ax.fill_between(xs, low, high, color=col, alpha=0.5, label=lab)
        ax.plot(xs, low, color='k', lw=0.25, alpha=0.1)#, label=lab)
        ax.plot(xs, high, color='k', lw=0.25, alpha=0.1)
        low = np.percentile(lines['cos_tilt_2'], 5, axis=0)
        high = np.percentile(lines['cos_tilt_2'], 95, axis=0)
        ax.fill_between(xs, low, high, color=col, alpha=0.1)
        ax.plot(xs, low, color='k', lw=0.05, alpha=0.05)#, label=lab)
        ax.plot(xs, high, color='k', lw=0.05, alpha=0.05)
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
for ps, lab, c in zip([ct1_pdfs, ct2_pdfs], ['cos_tilt_1', 'cos_tilt_2'],  ['tab:red', 'tab:orange']):
    low = np.percentile(ps, 35, axis=0)
    high = np.percentile(ps, 65, axis=0)#med = np.median(ps, axis=0)
    ax.plot(xs, low, color='k', lw=0.25, alpha=0.1)#, ls='--')
    ax.plot(xs, high,color='k', lw=0.25, alpha=0.1)#, ls='--')
    #ax.plot(xs, med, color=c, lw=4, alpha=0.75, label=f'MSpline-{lab}')
    ax.fill_between(xs, low, high, color=c, alpha=0.5, label='MSpline')
    low = np.percentile(ps, 5, axis=0)
    high = np.percentile(ps, 95, axis=0)
    #for _ in range(1000):
    #    idx = np.random.choice(ps.shape[0])
    #    ax.plot(xs, ps[idx], color='k', lw=0.035, alpha=0.035)
    ax.plot(xs, low, color='k', lw=0.05, alpha=0.05)#, ls='--')
    ax.plot(xs, high,color='k', lw=0.05, alpha=0.05)#, ls='--')
    #ax.plot(xs, med, color=c, lw=4, alpha=0.75, label=f'MSpline-{lab}')
    ax.fill_between(xs, low, high, color=c, alpha=0.1)
    if max(high) > maxy:
        maxy = max(high)

ax.set_xlabel(r'$\cos{\theta}$', fontsize=18)
ax.set_ylabel(r'$p(\cos{\theta})$', fontsize=18)

ax.set_xlim(xmin, xmax)

ax.set_ylim(0, maxy)
ax.legend(frameon=False, fontsize=14);
ax.grid(True, which="major", ls=":")
ax.tick_params(labelsize=14)
    
plt.title(f'GWTC-3: MSpline Independent Spin Tilt (12 knots)', fontsize=18);
fig.tight_layout()
plt.savefig(paths.figures / 'ind_spintilt.pdf', dpi=300);