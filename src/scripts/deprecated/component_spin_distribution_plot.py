#!/usr/bin/env python

import paths
import numpy as np
import matplotlib.pyplot as plt
import deepdish as dd

def load_mag_ppd():
    datadict = dd.io.load(paths.data / 'mspline_60m1_12ind_compspins_powerlaw_q_z_ppds.h5')
    return datadict['dRda1'], datadict['dRda2'], datadict['mags'], datadict['mags']

def load_tilt_ppd():
    datadict = dd.io.load(paths.data / 'mspline_60m1_12ind_compspins_powerlaw_q_z_ppds.h5')
    return datadict['dRdct1'], datadict['dRdct2'], datadict['tilts'], datadict['tilts']


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
        ax.plot(xs, high, color='k', lw=0.1, alpha=0.1)
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
        ax.plot(xs, high,  color='k', lw=0.1, alpha=0.1)#, label=lab)
    else:
        med = np.median(lines['cos_tilt_2'], axis=0)
        low = np.percentile(lines['cos_tilt_2'], 5, axis=0)
        high = np.percentile(lines['cos_tilt_2'], 95, axis=0)
        ax.plot(xs, med, color=col, lw=5, alpha=0.5, label=lab)
        ax.fill_between(xs, low, high, color=col, alpha=0.3)
        ax.plot(xs, low, color='k', lw=0.1, alpha=0.1)#, label=lab)
        ax.plot(xs, high, color='k', lw=0.1, alpha=0.1)
    return ax

figx, figy = 16, 10
fig, axs = plt.subplots(nrows=2, ncols=2, sharey='row', sharex='row', figsize=(figx,figy))

xmin, xmax = 0, 1
a1_pdfs, a2_pdfs, a1s, a2s = load_mag_ppd()
maxy=0
for i in range(len(a1_pdfs)):
    a1_pdfs[i] /= np.trapz(a1_pdfs[i], a1s)
    a2_pdfs[i] /= np.trapz(a2_pdfs[i], a2s)

for ax, ps, lab in zip([axs[0,0], axs[0,1]], [a1_pdfs, a2_pdfs], ['a1', 'a2']):
    med = np.median(ps, axis=0)
    low = np.percentile(ps, 5, axis=0)
    high = np.percentile(ps, 95, axis=0)
    #ax.fill_between(a1s, low, high, color='k', alpha=0.2)
    for _ in range(1000):
        idx = np.random.choice(ps.shape[0])
        ax.plot(a1s, ps[idx], color='k', lw=0.035, alpha=0.035)
    ax.plot(a1s, low, color='k', lw=0.1, alpha=0.1)
    ax.plot(a1s, high, color='k', lw=0.1, alpha=0.1)
    ax.plot(a1s, med, color='tab:red', lw=5, alpha=0.75, label='MSpline')
    ax.fill_between(a1s, low, high, color='tab:red', alpha=0.15)
    ax = plot_o3b_spinmag(ax,'o1o2o3_mass_c_iid_mag_iid_tilt_powerlaw_redshift_magnitude_data.h5', a1=lab=='a1', lab='PLPeak', col='tab:blue')
    #ax = plot_o3b_spinmag(ax,'spline_20n_mass_m_iid_mag_iid_tilt_powerlaw_redshift_magnitude_data.h5', a1=lab=='a1', lab='PLSpline', col='tab:orange')

    ax.legend(frameon=False, fontsize=14);
    if lab == 'a1':
        ax.set_xlabel(r'$a_1$', fontsize=18)
        ax.set_ylabel(r'$p(a_1)$', fontsize=18)
    else:
        ax.set_xlabel(r'$a_2$', fontsize=18)
        ax.set_ylabel(r'$p(a_2)$', fontsize=18)
    ax.set_xlim(xmin, xmax)
    if max(high) > maxy:
        maxy = max(high)
    ax.set_ylim(0, maxy)
    ax.legend(frameon=False, fontsize=14);
    ax.grid(True, which="major", ls=":")
    ax.tick_params(labelsize=14)

ct1_pdfs, ct2_pdfs, xs, xs = load_tilt_ppd()
for i in range(len(ct1_pdfs)):
    ct1_pdfs[i] /= np.trapz(ct1_pdfs[i], xs)
    ct2_pdfs[i] /= np.trapz(ct2_pdfs[i], xs)
xmin=-1
maxy=0
for ax, ps, lab in zip([axs[1,0], axs[1,1]], [ct1_pdfs, ct2_pdfs], ['cos_tilt_1', 'cos_tilt_2']):
    med = np.median(ps, axis=0)
    low = np.percentile(ps, 5, axis=0)
    high = np.percentile(ps, 95, axis=0)
    for _ in range(1000):
        idx = np.random.choice(ps.shape[0])
        ax.plot(xs, ps[idx], color='k', lw=0.035, alpha=0.035)
    ax.plot(xs, low, color='k', lw=0.1, alpha=0.1)#, ls='--')
    ax.plot(xs, high,color='k', lw=0.1, alpha=0.1)#, ls='--')
    ax.plot(xs, med, color='tab:red', lw=5, alpha=0.75, label='MSpline')
    ax.fill_between(xs, low, high, color='tab:red', alpha=0.15)
    ax = plot_o3b_spintilt(ax,'o1o2o3_mass_c_iid_mag_iid_tilt_powerlaw_redshift_orientation_data.h5', ct1=lab=='cos_tilt_1', lab='PLPeak', col='tab:blue')
    #ax = plot_o3b_spintilt(ax,'spline_20n_mass_m_iid_mag_iid_tilt_powerlaw_redshift_orientation_data.h5', ct1=lab=='cos_tilt_1', lab='PLSpline', col='tab:orange')

    ax.legend(frameon=False, fontsize=14);
    if lab == 'cos_tilt_1':
        ax.set_xlabel(r'$\cos{\theta_1}$', fontsize=18)
        ax.set_ylabel(r'$p(\cos{\theta_1})$', fontsize=18)
    else:
        ax.set_xlabel(r'$\cos{\theta_2}$', fontsize=18)
        ax.set_ylabel(r'$p(\cos{\theta_2})$', fontsize=18)
    ax.set_xlim(xmin, xmax)
    if max(high) > maxy:
        maxy = max(high)
    ax.set_ylim(0, maxy)
    ax.legend(frameon=False, fontsize=14);
    ax.grid(True, which="major", ls=":")
    ax.tick_params(labelsize=14)

plt.suptitle(f'GWTC-3: MSpline Independent Spin Mag (12 knots) + Independent Spin Tilt (12 knots)', fontsize=18);
fig.tight_layout()
plt.savefig(paths.figures / 'component_spin_distribution_plot.pdf', dpi=300);