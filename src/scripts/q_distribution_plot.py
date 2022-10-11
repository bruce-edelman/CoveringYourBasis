#!/usr/bin/env python

import paths
import numpy as np
import matplotlib.pyplot as plt
from utils import plot_mean_and_90CI, load_mass_ppd, load_o3b_paper_run_masspdf
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

figx, figy = 7, 5
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(figx,figy))


for ax, xs, ps, lab in zip([axs], [qs], [q_pdfs], ['q']):
    ax = plot_o3b_res(ax,'o1o2o3_mass_c_iid_mag_iid_tilt_powerlaw_redshift_mass_data.h5', m1=False, lab='PL+Peak', col='tab:blue', bounds=False)#, fill_alpha=0.15)
    ax = plot_o3b_res(ax,'spline_20n_mass_m_iid_mag_iid_tilt_powerlaw_redshift_mass_data.h5', m1=False, lab='PL+Spline', col='tab:green', bounds=False)#, fill_alpha=0.15)
    ax = plot_mean_and_90CI(ax, qs, q_pdfs, color='tab:red', label='BSpline', bounds=True)
    ax.legend(frameon=False, fontsize=14);
    ax.set_xlabel(r'$q$', fontsize=18)
    ax.set_ylabel(r'$p(q)$', fontsize=18)
    ax.grid(True, which="major", ls=":")
    ax.tick_params(labelsize=14)
    


axs.set_yscale('log')
axs.set_ylim(1e-2, 1e1)
axs.grid(True, which="major", ls=":")
axs.set_xlim(0.03, 1)
plt.title(f'GWTC-3: BBH Mass Ratio Distribution', fontsize=18);
fig.tight_layout()
plt.savefig(paths.figures / 'q_distribution_plot.pdf', dpi=300);