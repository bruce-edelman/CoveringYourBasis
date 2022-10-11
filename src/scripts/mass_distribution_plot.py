#!/usr/bin/env python

import paths
import numpy as np
import matplotlib.pyplot as plt
from utils import plot_mean_and_90CI, load_mass_ppd, load_o3b_paper_run_masspdf
from matplotlib.ticker import ScalarFormatter
import deepdish as dd


def plot_o3b_res(ax, fi, m1=True, col='tab:blue', lab='PP', bounds=False, fill_alpha=0.08):
    plpeak_mpdfs, plpeak_qpdfs, plpeak_ms, plpeak_qs = load_o3b_paper_run_masspdf(paths.data / fi)
    if m1:
        plot_mean_and_90CI(ax, plpeak_ms, plpeak_mpdfs, color=col, label=lab, bounds=bounds, fill_alpha=fill_alpha)
    else:
        plot_mean_and_90CI(ax, plpeak_qs, plpeak_qpdfs, color=col, label=lab, bounds=bounds, fill_alpha=fill_alpha)
    return ax


mmin = 6
mmax = 100
ms, m_pdfs, qs, q_pdfs = load_mass_ppd()
for jj in range(len(m_pdfs)):
    m_pdfs[jj,:] /= np.trapz(m_pdfs[jj,:], ms)
    q_pdfs[jj,:] /= np.trapz(q_pdfs[jj,:], qs)

figx, figy = 14, 5
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(figx,figy))

def load_plsplinemass_ppd():
    datadict = dd.io.load(paths.data / 'plbspline_38n_mspline_12n_iid_compsins_8chains_ppds.h5')
    return datadict['m1s'], datadict['dRdm1'], datadict['qs'], datadict['dRdq']

def load_bsplinemass_ppd():
    datadict = dd.io.load(paths.data / 'bsplines_64m1_18q_iid18mag_iid16tilt_pl16z_ppds.h5')
    return datadict['m1s'], datadict['dRdm1'], datadict['qs'], datadict['dRdq']

#plspl_ms, plspl_mpdfs, plspl_qs, plspl_qpdfs = load_plsplinemass_ppd()
bspl_ms, bspl_mpdfs, bspl_qs, bspl_qpdfs = load_bsplinemass_ppd()

for ax, xs, ps, lab in zip([axs], [ms], [m_pdfs], ['m1']):
    ax = plot_o3b_res(ax,'o1o2o3_mass_c_iid_mag_iid_tilt_powerlaw_redshift_mass_data.h5', lab='PL+Peak', col='tab:blue', bounds=False)#, fill_alpha=0.15)
    ax = plot_o3b_res(ax,'spline_20n_mass_m_iid_mag_iid_tilt_powerlaw_redshift_mass_data.h5', lab='PL+Spline', col='tab:green', bounds=False)#, fill_alpha=0.15)
    #ax = plot_mean_and_90CI(ax, xs, ps, color='tab:red', label='MSpline', fill_alpha=0.125)
    #ax = plot_mean_and_90CI(ax, plspl_ms, plspl_mpdfs, color='tab:purple', label='PL+BSpline', bounds=False)#, fill_alpha=0.125)
    ax = plot_mean_and_90CI(ax, bspl_ms, bspl_mpdfs, color='tab:red', label='BSpline', bounds=True)
    ax.legend(frameon=False, fontsize=14);
    ax.set_xlabel(r'$m_1 \,\,[M_\odot]$', fontsize=18)
    ax.set_ylabel(r'$p(m_1) \,\,[M_\odot^{-1}]$', fontsize=18)
    ax.grid(True, which="major", ls=":")
    ax.tick_params(labelsize=14)
    


axs.set_yscale('log')
axs.set_xscale('log')
axs.set_ylim(1e-5, 1e0)
logticks = np.array([6,8,10,20,40,70,100])
axs.set_xticks(logticks)
axs.get_xaxis().set_major_formatter(ScalarFormatter())
axs.grid(True, which="major", ls=":")
axs.set_xlim(mmin+0.5, mmax)
plt.title(f'GWTC-3: BBH Primary Mass Distribution', fontsize=18);
fig.tight_layout()
plt.savefig(paths.figures / 'mass_distribution_plot.pdf', dpi=300);