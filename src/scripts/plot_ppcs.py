#!/usr/bin/env python

import paths
import numpy as np
import matplotlib.pyplot as plt
import deepdish as dd


params = [
    "mass_1",
    "mass_ratio",
    "redshift",
]

param_latex = {
    "mass_1": r'$m_1$',
    "mass_ratio": r'$q$',
    "redshift": r'$z$',
}

nplot = len(params)
figx, figy = 12,5
fig, axs = plt.subplots(nrows=1, ncols=nplot, sharey='row', figsize=(figx,figy))

Nobs = 69
po = dd.io.load(paths.data / 'bsplines_64m1_18q_iid18mag_iid18tilt_pl18z_posterior_samples.h5')

for ax, param in zip(axs, params):
    observed = np.array([po[f"{param}_obs_event_{i}"] for i in range(Nobs)])
    synthetic = np.array([po[f"{param}_pred_event_{i}"] for i in range(Nobs)])
    
    if param == 'redshift':
        zmax = max([np.max(observed) , np.max(synthetic)])
        zmin = min([np.min(observed) , np.min(synthetic)])

        
    ax.fill_betweenx(
        y=np.linspace(0, 1, len(observed[:, 0])),
        x1=np.quantile(np.sort(observed, axis=0), 0.05, axis=1),
        x2=np.quantile(np.sort(observed, axis=0), 0.95, axis=1),
        color="k",
        alpha=0.25,
        label="Observed",
    )
    ax.fill_betweenx(
        y=np.linspace(0, 1, len(synthetic[:, 0])),
        x1=np.quantile(np.sort(synthetic, axis=0), 0.05, axis=1),
        x2=np.quantile(np.sort(synthetic, axis=0), 0.95, axis=1),
        color="tab:red",
        alpha=0.2,
        label="Predicted",
    )
    ax.plot(
        np.median(np.sort(synthetic, axis=0), axis=1),
        np.linspace(0, 1, len(synthetic[:, 0])),
        color="tab:red",
        alpha=0.5,
        lw=4, label='Median'
    )
    ax.set_xlim(
        min(np.min(synthetic), np.min(observed)),
        max(np.max(synthetic), np.max(observed)),
    )
    ax.tick_params(labelsize=13)
    ax.set_ylim(0, 1)
    ax.grid(which="both", ls=":", lw=1)
    ax.set_xlabel(param_latex[param], fontsize=18)
    if param == "mass_1":
        ax.set_xlim(5, 100)
        ax.set_xscale("log")
        ax.set_ylabel("Cumulative Probability", fontsize=16)
    elif param == "redshift":
        ax.set_xscale('log')
        ax.set_xlim(0.05, zmax)
    else:
        ax.set_xlim(0.05,1)
        ax.legend(frameon=False, fontsize=16)
        
plt.suptitle(f'GWTC-3: B-Spline Model Posterior Predictive Checks', fontsize=20);
fig.tight_layout()
plt.subplots_adjust(wspace=0.05)
plt.savefig(paths.figures / 'ppc_plot.pdf', dpi=300);