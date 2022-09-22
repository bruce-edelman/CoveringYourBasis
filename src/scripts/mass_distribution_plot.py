#!/usr/bin/env python

import paths
import numpy as np
import matplotlib.pyplot as plt
import deepdish as dd

def load_ppd():
    datadict = dd.io.load(paths.data / 'mspline_60m1_14iid_compspins_powerlaw_q_z_ppds.h5')
    return datadict['m1s'], datadict['dRdm1'], datadict['qs'], datadict['dRdq']


def load_o3b_paper_run_masspdf(filename):
    """
    Generates a plot of the PPD and X% credible region for the mass distribution,
    where X=limits[1]-limits[0]
    """
    mass_1 = np.linspace(2, 100, 1000)
    mass_ratio = np.linspace(0.1, 1, 500)
        
    # load in the traces. 
    # Each entry in lines is p(m1 | Lambda_i) or p(q | Lambda_i)
    # where Lambda_i is a single draw from the hyperposterior
    # The ppd is a 2D object defined in m1 and q
    with open(filename, 'r') as _data:
        _data = dd.io.load(filename)
        marginals = _data["lines"]
    for ii in range(len(marginals['mass_1'])):
        marginals['mass_1'][ii] /= np.trapz(marginals['mass_1'][ii], mass_1)
        marginals['mass_ratio'][ii] /= np.trapz(marginals['mass_ratio'][ii], mass_ratio)
    return marginals['mass_1'], marginals['mass_ratio'], mass_1, mass_ratio


def plot_o3b_res(ax, fi, m1=False, col='tab:blue', lab='PP'):
    plpeak_mpdfs, plpeak_qpdfs, plpeak_ms, plpeak_qs = load_o3b_paper_run_masspdf(paths.data / fi)
    if m1:
        med = np.median(plpeak_mpdfs, axis=0)
        low = np.percentile(plpeak_mpdfs, 5, axis=0)
        high = np.percentile(plpeak_mpdfs, 95, axis=0)
        ax.plot(plpeak_ms, med, color=col, lw=5, alpha=0.5, label=lab)
        ax.fill_between(plpeak_ms, low, high, color=col, alpha=0.3)
        ax.plot(plpeak_ms, low, color='k', lw=0.1, alpha=0.1)#, label=lab)
        ax.plot(plpeak_ms, high, color='k', lw=0.1, alpha=0.1)
    else:
        med = np.median(plpeak_qpdfs, axis=0)
        low = np.percentile(plpeak_qpdfs, 5, axis=0)
        high = np.percentile(plpeak_qpdfs, 95, axis=0)
        ax.plot(plpeak_qs, med, color=col, lw=5, alpha=0.5, label=lab)
        ax.fill_between(plpeak_qs, low, high, color=col, alpha=0.3)
        ax.plot(plpeak_qs, low, color='k', lw=0.1, alpha=0.1)#, label=lab)
        ax.plot(plpeak_qs, high, color='k', lw=0.1, alpha=0.1)#, label=lab)

    return ax

mmin = 6.5
mmax = 100
ms, m_pdfs, qs, q_pdfs = load_ppd()
for jj in range(len(m_pdfs)):
    m_pdfs[jj,:] /= np.trapz(m_pdfs[jj,:], ms)
    q_pdfs[jj,:] /= np.trapz(q_pdfs[jj,:], qs)

figx, figy = 14, 5
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(figx,figy))

for ax, xs, ps, lab in zip([axs], [ms], [m_pdfs], ['m1']):
    med = np.median(ps, axis=0)
    low = np.percentile(ps, 5, axis=0)
    high = np.percentile(ps, 95, axis=0)
    for _ in range(1000):
        idx = np.random.choice(ps.shape[0])
        ax.plot(xs, ps[idx], color='k', lw=0.025, alpha=0.02)    
    ax.plot(xs, low,color='k', lw=0.1, alpha=0.1)#, ls='--')#, label='MSpline')
    ax.plot(xs, high, color='k', lw=0.1, alpha=0.1)#, ls='--')#, label='MSpline')
    ax.plot(xs, med, color='tab:red', lw=5, alpha=0.75, label='MSpline')
    ax.fill_between(xs, low, high, color='tab:red', alpha=0.15)

    #ax = plot_o3b_res(ax,'o1o2o3_mass_c_iid_mag_iid_tilt_powerlaw_redshift_mass_data.h5', m1=lab=='m1', lab='PLPeak', col='tab:blue')
    ax = plot_o3b_res(ax,'spline_20n_mass_m_iid_mag_iid_tilt_powerlaw_redshift_mass_data.h5', m1=lab=='m1', lab='PLSpline', col='tab:blue')

    ax.legend(frameon=False, fontsize=14);
    if lab == 'm1':
        ax.set_xlabel(r'$m_1 \,\,[M_\odot]$', fontsize=18)
        ax.set_ylabel(r'$p_{MS}(m_1) \,\,[M_\odot^{-1}]$', fontsize=18)
    else:
        ax.set_xlabel(r'$q$', fontsize=18)
        ax.set_ylabel(r'$p_{MS}(q)$', fontsize=18)
    ax.grid(True, which="major", ls=":")
    ax.tick_params(labelsize=14)

logticks = np.array([5,10,50,100])
axs.set_xticks(logticks)
axs.grid(True, which="major", ls=":")

axs.set_yscale('log')
axs.set_xscale('log')
axs.set_xlim(mmin, mmax)
axs.set_ylim(8e-5, 4e-1)
plt.title(f'GWTC-3: MSpline Primary Mass Distribution', fontsize=18);
fig.tight_layout()
plt.savefig(paths.figures / 'mass_distribution_plot.pdf', dpi=300);