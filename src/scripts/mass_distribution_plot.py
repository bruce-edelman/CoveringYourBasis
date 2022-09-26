#!/usr/bin/env python

import paths
import numpy as np
import matplotlib.pyplot as plt
from utils import plot_mean_and_90CI, load_mass_ppd, load_o3b_paper_run_masspdf
from matplotlib.ticker import ScalarFormatter


def plot_o3b_res(ax, fi, m1=True, col='tab:blue', lab='PP', bounds=False):
    plpeak_mpdfs, plpeak_qpdfs, plpeak_ms, plpeak_qs = load_o3b_paper_run_masspdf(paths.data / fi)
    if m1:
        plot_mean_and_90CI(ax, plpeak_ms, plpeak_mpdfs, color=col, label=lab, bounds=bounds)
    else:
        plot_mean_and_90CI(ax, plpeak_qs, plpeak_qpdfs, color=col, label=lab, bounds=bounds)
    return ax


mmin = 6.5
mmax = 100
ms, m_pdfs, qs, q_pdfs = load_mass_ppd()
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