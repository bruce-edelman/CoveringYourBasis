#!/usr/bin/env python

import paths
import numpy as np
import matplotlib.pyplot as plt
import deepdish as dd
from utils import plot_mean_and_90CI
from matplotlib.ticker import ScalarFormatter


def load_ppd():
    datadict = dd.io.load(paths.data / 'mspline_50m1_16iid_compspins_smoothprior_powerlaw_q_z_ppds.h5')
    return datadict['m1s'], datadict['dRdm1'], datadict['qs'], datadict['dRdq']


def load_o3b_paper_run_m1ppd(filename):
    mass_1 = np.linspace(2, 100, 1000)
    mass_ratio = np.linspace(0.1, 1, 500)
    # load in the traces. 
    # Each entry in lines is p(m1 | Lambda_i) or p(q | Lambda_i)
    # where Lambda_i is a single draw from the hyperposterior
    # The ppd is a 2D object defined in m1 and q
    with open(filename, 'r') as _data:
        _data = dd.io.load(filename)
        ppd = _data["ppd"]
    m1ppd = np.trapz(ppd, x=mass_ratio, axis=0)
    return mass_1, m1ppd / np.trapz(m1ppd, x=mass_1)

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

def plot_o3b_m1_ppd(ax, fi, col='tab:blue', lab='PLPeak'):
    ms, ppd = load_o3b_paper_run_m1ppd(paths.data / fi)
    ax.plot(ms, ppd, color=col, label=lab, alpha=0.75, lw=4)
    return ax

def plot_o3b_res(ax, fi, m1=True, col='tab:blue', lab='PP', bounds=False):
    plpeak_mpdfs, plpeak_qpdfs, plpeak_ms, plpeak_qs = load_o3b_paper_run_masspdf(paths.data / fi)
    if m1:
        plot_mean_and_90CI(ax, plpeak_ms, plpeak_mpdfs, color=col, label=lab, bounds=bounds)
    else:
        plot_mean_and_90CI(ax, plpeak_qs, plpeak_qpdfs, color=col, label=lab, bounds=bounds)
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
    ax = plot_o3b_res(ax,'o1o2o3_mass_c_iid_mag_iid_tilt_powerlaw_redshift_mass_data.h5', lab='PLPeak', col='tab:blue')
    ax = plot_o3b_res(ax,'spline_20n_mass_m_iid_mag_iid_tilt_powerlaw_redshift_mass_data.h5', lab='PLSpline', col='tab:green')
    ax = plot_mean_and_90CI(ax, xs, ps, color='tab:red', label='MSpline')
    ax.legend(frameon=False, fontsize=14);
    ax.set_xlabel(r'$m_1 \,\,[M_\odot]$', fontsize=18)
    ax.set_ylabel(r'$p_{MS}(m_1) \,\,[M_\odot^{-1}]$', fontsize=18)
    ax.grid(True, which="major", ls=":")
    ax.tick_params(labelsize=14)

axs.set_yscale('log')
axs.set_xscale('log')
axs.set_xlim(mmin, mmax)
axs.set_ylim(1e-4, 3.5e-1)
logticks = np.array([6,8,10,20,40,70,100])
axs.set_xticks(logticks)
axs.get_xaxis().set_major_formatter(ScalarFormatter())
axs.grid(True, which="major", ls=":")
plt.title(f'GWTC-3: MSpline Primary Mass Distribution', fontsize=18);
fig.tight_layout()
plt.savefig(paths.figures / 'mass_distribution_plot.pdf', dpi=300);