#!/usr/bin/env python

import paths
import numpy as np
import matplotlib.pyplot as plt
import deepdish as dd


params = [
    "mass_1",
    "mass_ratio",
    "redshift",
    "a",
    "cos_tilt",
]

nplot = len(params)
figx, figy = 20, 4
fig, axs = plt.subplots(nrows=1, ncols=nplot, sharey='row', figsize=(figx,figy))

Nobs = 69
po = dd.io.load(paths.data / 'bsplines_64m1_18q_iid18mag_iid16tilt_pl16z_posterior_samples.h5')

for ax, param in zip(axs, params):
    if param == 'a' or param == 'cos_tilt':
        observed1 = np.array([po[f"{param}_1_obs_event_{i}"] for i in range(Nobs)])
        synthetic1 = np.array([po[f"{param}_1_pred_event_{i}"] for i in range(Nobs)])
        observed2 = np.array([po[f"{param}_2_obs_event_{i}"] for i in range(Nobs)])
        synthetic2 = np.array([po[f"{param}_2_pred_event_{i}"] for i in range(Nobs)])
        observed = np.concatenate([observed1, observed2]).reshape((2*observed1.shape[0],Nobs))
        synthetic = np.concatenate([synthetic1, synthetic2]).reshape((2*synthetic1.shape[0],Nobs))
    else:    
        observed = np.array([po[f"{param}_obs_event_{i}"] for i in range(Nobs)])
        synthetic = np.array([po[f"{param}_pred_event_{i}"] for i in range(Nobs)])
    if param == 'redshift':
        zmax = max([max(observed), max(synthetic)])
        
    ax.fill_betweenx(
        y=np.linspace(0, 1, len(observed[:, 0])),
        x1=np.quantile(np.sort(observed, axis=0), 0.05, axis=1),
        x2=np.quantile(np.sort(observed, axis=0), 0.95, axis=1),
        color="tab:blue",
        alpha=0.8,
        label="Observed",
    )
    ax.plot(
        np.quantile(np.sort(observed, axis=0), 0.05, axis=1),
        np.linspace(0, 1, len(observed[:, 0])),
        color="k",
        alpha=0.25,
        lw=0.15,
    )
    ax.plot(
        np.quantile(np.sort(observed, axis=0), 0.95, axis=1),
        np.linspace(0, 1, len(observed[:, 0])),
        color="k",
        alpha=0.25,
        lw=0.15,
    )

    ax.fill_betweenx(
        y=np.linspace(0, 1, len(synthetic[:, 0])),
        x1=np.quantile(np.sort(synthetic, axis=0), 0.05, axis=1),
        x2=np.quantile(np.sort(synthetic, axis=0), 0.95, axis=1),
        color="tab:blue",
        alpha=0.3,
        label="Predicted",
    )
    ax.plot(
        np.quantile(np.sort(synthetic, axis=0), 0.05, axis=1),
        np.linspace(0, 1, len(synthetic[:, 0])),
        color="k",
        alpha=0.25,
        lw=0.15,
    )
    ax.plot(
        np.quantile(np.sort(synthetic, axis=0), 0.95, axis=1),
        np.linspace(0, 1, len(synthetic[:, 0])),
        color="k",
        alpha=0.25,
        lw=0.15,
    )
    ax.plot(
        np.median(np.sort(synthetic, axis=0), axis=1),
        np.linspace(0, 1, len(synthetic[:, 0])),
        color="tab:blue",
        alpha=0.9,
        lw=4,
    )
    ax.legend(loc="upper left")
    ax.set_xlim(
        min(np.min(synthetic), np.min(observed)),
        max(np.max(synthetic), np.max(observed)),
    )

    ax.set_ylim(0, 1)
    ax.grid(which="both", ls=":", lw=1)
    ax.set_ylabel("Cumulative Probability")
    ax.set_xlabel(param)

    if param == "mass_1":
        ax.set_xlim(6.5, 100)
        ax.set_xscale("log")
    elif param == 'cos_tilt' or param == 'chi_eff':
        ax.set_xlim(-1, 1)
    elif param == "redshift":
        ax.set_xlim(0, zmax)
    else:
        ax.set_xlim(0, 1)
        
plt.suptitle(f'GWTC-3: Basis Spline Posterior Predictive Checks', fontsize=18);
fig.tight_layout()
plt.savefig(paths.figures / 'ppc_plot.pdf', dpi=300);