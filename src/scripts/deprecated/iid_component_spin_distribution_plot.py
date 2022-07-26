#!/usr/bin/env python

import paths
import numpy as np
import matplotlib.pyplot as plt
import deepdish as dd

def load_mag_ppd():
    datadict = dd.io.load(paths.data / 'mspline_60m1_12iid_compspins_powerlaw_q_z_ppds.h5')
    return datadict['mags'], datadict['dRda']

def load_tilt_ppd():
    datadict = dd.io.load(paths.data / 'mspline_60m1_12iid_compspins_powerlaw_q_z_ppds.h5')
    return datadict['tilts'], datadict['dRdct']


def plot_o3b_spinmag(ax, fi, a1=False, col='tab:blue', lab='PP'):
    xs = np.linspace(0, 1, 1000)
    
    _data = dd.io.load(paths.data / fi)
    lines = dict()
    for key in _data["lines"].keys():
        lines[key] = _data["lines"][key][()]
        for ii in range(len(lines[key])):
            lines[key][ii] /= np.trapz(lines[key][ii], xs)
    if a1:
        med = np.median(lines['a_1'], axis=0)
        low = np.percentile(lines['a_1'], 5, axis=0)
        high = np.percentile(lines['a_1'], 95, axis=0)
        ax.plot(xs, med, color=col, lw=5, alpha=0.5, label=lab)
        ax.fill_between(xs, low, high, color=col, alpha=0.3)
        ax.plot(xs, low, color='k', lw=0.1, alpha=0.1)#, label=lab)
        ax.plot(xs, high, color='k', lw=0.1, alpha=0.1)#, label=lab)
    else:
        med = np.median(lines['a_2'], axis=0)
        low = np.percentile(lines['a_2'], 5, axis=0)
        high = np.percentile(lines['a_2'], 95, axis=0)
        ax.plot(xs, med, color=col, lw=5, alpha=0.5, label=lab)
        ax.fill_between(xs, low, high, color=col, alpha=0.3)
        ax.plot(xs, low, color='k', lw=0.1, alpha=0.1)#, label=lab)
        ax.plot(xs, high,color='k', lw=0.1, alpha=0.1)
    return ax


def plot_o3b_spintilt(ax, fi,ct1=False, col='tab:blue', lab='PP'):
    xs = np.linspace(-1, 1, 1000)
    _data = dd.io.load(paths.data / fi)
    lines = dict()
    for key in _data["lines"].keys():
        lines[key] = _data["lines"][key][()]
        for ii in range(len(lines[key])):
            lines[key][ii] /= np.trapz(lines[key][ii], xs)
    if ct1:
        med = np.median(lines['cos_tilt_1'], axis=0)
        low = np.percentile(lines['cos_tilt_1'], 5, axis=0)
        high = np.percentile(lines['cos_tilt_1'], 95, axis=0)
        ax.plot(xs, med, color=col, lw=5, alpha=0.5, label=lab)
        ax.fill_between(xs, low, high, color=col, alpha=0.3)
        ax.plot(xs, low, color='k', lw=0.1, alpha=0.1)#, label=lab)
        ax.plot(xs, high, color='k', lw=0.1, alpha=0.1)#, label=lab)
    else:
        med = np.median(lines['cos_tilt_2'], axis=0)
        low = np.percentile(lines['cos_tilt_2'], 5, axis=0)
        high = np.percentile(lines['cos_tilt_2'], 95, axis=0)
        ax.plot(xs, med, color=col, lw=5, alpha=0.5, label=lab)
        ax.fill_between(xs, low, high, color=col, alpha=0.3)
        ax.plot(xs, low, color='k', lw=0.1, alpha=0.1)#, label=lab)
        ax.plot(xs, high, color='k', lw=0.1, alpha=0.1)
    return ax

figx, figy = 16, 5
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(figx,figy))

xmin, xmax = 0, 1
mags, a_pdfs = load_mag_ppd()
for jj in range(len(a_pdfs)):
    a_pdfs[jj,:] /= np.trapz(a_pdfs[jj,:], mags)
ax = axs[0]
med = np.median(a_pdfs , axis=0)
low = np.percentile(a_pdfs , 5, axis=0)
high = np.percentile(a_pdfs , 95, axis=0)
for _ in range(1000):
    idx = np.random.choice(a_pdfs.shape[0])
    ax.plot(mags, a_pdfs[idx], color='k', lw=0.05, alpha=0.045)    
ax.plot(mags, low, color='k', lw=0.1, alpha=0.1)#, ls='--')
ax.plot(mags, high, color='k', lw=0.1, alpha=0.1)#, ls='--')
ax.plot(mags, med, color='tab:red', lw=5, alpha=0.75, label='MSpline')
ax.fill_between(mags, low, high, color='tab:red', alpha=0.15)

ax = plot_o3b_spinmag(ax,'o1o2o3_mass_c_iid_mag_iid_tilt_powerlaw_redshift_magnitude_data.h5', a1=True, lab='PLPeak', col='tab:blue')
#ax = plot_o3b_spinmag(ax,'spline_20n_mass_m_iid_mag_iid_tilt_powerlaw_redshift_magnitude_data.h5', a1=True, lab='PLSpline', col='tab:orange')

ax.legend(frameon=False, fontsize=14);
ax.set_xlabel(r'$a$', fontsize=18)
ax.set_ylabel(r'$p(a)$', fontsize=18)

ax.set_xlim(xmin, xmax)
ax.set_ylim(0, max(high))
ax.legend(frameon=False, fontsize=14);
ax.grid(True, which="major", ls=":")
ax.tick_params(labelsize=14)

xs, ct_pdfs = load_tilt_ppd()
for jj in range(len(ct_pdfs)):
    ct_pdfs[jj,:] /= np.trapz(ct_pdfs[jj,:], xs)
xmin=-1
ax = axs[1]
med = np.median(ct_pdfs , axis=0)
low = np.percentile(ct_pdfs , 5, axis=0)
high = np.percentile(ct_pdfs , 95, axis=0)
for _ in range(1000):
    idx = np.random.choice(ct_pdfs.shape[0])
    ax.plot(xs, ct_pdfs[idx], color='k', lw=0.05, alpha=0.045)   
ax.plot(xs, med, color='tab:red', lw=5, alpha=0.75, label='MSpline')
ax.plot(xs, color='k', lw=0.1, alpha=0.1)
ax.plot(xs, color='k', lw=0.1, alpha=0.1)
ax.fill_between(xs, low, high, color='tab:red', alpha=0.15)#, label='90% CI')

ax = plot_o3b_spintilt(ax,'o1o2o3_mass_c_iid_mag_iid_tilt_powerlaw_redshift_orientation_data.h5', ct1=True, lab='PLPeak', col='tab:blue')
#ax = plot_o3b_spintilt(ax,'spline_20n_mass_m_iid_mag_iid_tilt_powerlaw_redshift_orientation_data.h5', ct1=True, lab='PLSpline', col='tab:orange')

ax.legend(frameon=False, fontsize=14);
ax.set_xlabel(r'$\cos{\theta}$', fontsize=18)
ax.set_ylabel(r'$p(\cos{\theta})$', fontsize=18)

ax.set_xlim(xmin, xmax)
ax.legend(frameon=False, fontsize=14);
ax.grid(True, which="major", ls=":")
ax.tick_params(labelsize=14)
ax.set_ylim(0, max(high))

plt.suptitle(f'GWTC-3: MSpline IID Spin Mag (12 knots) IID Spin Tilt (12 knots)', fontsize=18);
fig.tight_layout()
plt.savefig(paths.figures / 'iid_component_spin_distribution_plot.pdf', dpi=300);