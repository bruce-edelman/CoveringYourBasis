#!/usr/bin/env python

import paths
import numpy as np
import matplotlib.pyplot as plt
import deepdish as dd

def load_tilt_ppd():
    datadict = dd.io.load(paths.data / 'mspline_60m1_14iid_compspins_powerlaw_q_z_ppds.h5')
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

figx, figy = 7, 5
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(figx,figy))

xmin, xmax = -1, 1
xs, ct_pdfs = load_tilt_ppd()
for jj in range(len(ct_pdfs)):
    ct_pdfs[jj,:] /= np.trapz(ct_pdfs[jj,:], xs)
xmin=-1
ax = axs
ax = plot_o3b_spintilt(ax,'o1o2o3_mass_c_iid_mag_iid_tilt_powerlaw_redshift_orientation_data.h5', ct1=True, lab='Default', col='tab:blue')
low = np.percentile(ct_pdfs , 35, axis=0)
high = np.percentile(ct_pdfs , 65, axis=0)
ax.plot(xs, color='k', lw=0.25, alpha=0.1)
ax.plot(xs, color='k', lw=0.25, alpha=0.1)
ax.fill_between(xs, low, high, color='tab:red', alpha=0.5, label='MSpline')#, label='90% CI')
low = np.percentile(ct_pdfs , 5, axis=0)
high = np.percentile(ct_pdfs , 95, axis=0)
#for _ in range(1000):
#    idx = np.random.choice(ct_pdfs.shape[0])
#    ax.plot(xs, ct_pdfs[idx], color='k', lw=0.025, alpha=0.025)   
ax.plot(xs, color='k', lw=0.05, alpha=0.05)
ax.plot(xs, color='k', lw=0.05, alpha=0.05)
ax.fill_between(xs, low, high, color='tab:red', alpha=0.1)#, label='90% CI')


ax.legend(frameon=False, fontsize=14);
ax.set_xlabel(r'$\cos{\theta}$', fontsize=18)
ax.set_ylabel(r'$p(\cos{\theta})$', fontsize=18)

ax.set_xlim(xmin, xmax)
ax.legend(frameon=False, fontsize=14);
ax.grid(True, which="major", ls=":")
ax.tick_params(labelsize=14)
ax.set_ylim(0, max(high))

plt.suptitle(f'GWTC-3: MSpline IID Spin Tilt (12 knots)', fontsize=18);
fig.tight_layout()
plt.savefig(paths.figures / 'iid_spintilt.pdf', dpi=300);