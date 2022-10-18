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

param_latex = {
    "mass_1": r'$m_1$',
    "mass_ratio": r'$q$',
    "redshift": r'$z$',
    "a": r'$a$',
    "cos_tilt": r'$\cos{\theta}$',
}

nplot = len(params)
figx, figy = 18,5
fig, axs = plt.subplots(nrows=1, ncols=nplot, sharey='row', figsize=(figx,figy))

Nobs = 69
po = dd.io.load(paths.data / 'bsplines_64m1_18q_iid18mag_iid16tilt_pl16z_posterior_samples.h5')

for ax, param in zip(axs, params):
    if param == 'a' or param == 'cos_tilt':
        observed1 = np.array([po[f"{param}_1_obs_event_{i}"] for i in range(Nobs)])
        synthetic1 = np.array([po[f"{param}_1_pred_event_{i}"] for i in range(Nobs)])
        observed2 = np.array([po[f"{param}_2_obs_event_{i}"] for i in range(Nobs)])
        synthetic2 = np.array([po[f"{param}_2_pred_event_{i}"] for i in range(Nobs)])
        observed = np.concatenate([observed1, observed2]).reshape((Nobs,2*observed1.shape[1]))
        synthetic = np.concatenate([synthetic1, synthetic2]).reshape((Nobs,2*synthetic1.shape[1]))
    else:    
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
    #ax.plot(
    #    np.quantile(np.sort(observed, axis=0), 0.05, axis=1),
    #    np.linspace(0, 1, len(observed[:, 0])),
    #    color="k",
    #    alpha=0.25,
    #    lw=0.15,
    #)
    #ax.plot(
    #    np.quantile(np.sort(observed, axis=0), 0.95, axis=1),
    #    np.linspace(0, 1, len(observed[:, 0])),
    #    color="k",
    #    alpha=0.25,
    #    lw=0.15,
    #)

    ax.fill_betweenx(
        y=np.linspace(0, 1, len(synthetic[:, 0])),
        x1=np.quantile(np.sort(synthetic, axis=0), 0.05, axis=1),
        x2=np.quantile(np.sort(synthetic, axis=0), 0.95, axis=1),
        color="tab:red",
        alpha=0.2,
        label="Predicted",
    )
    #ax.plot(
    #    np.quantile(np.sort(synthetic, axis=0), 0.05, axis=1),
    #    np.linspace(0, 1, len(synthetic[:, 0])),
    #    color="k",
    #    alpha=0.25,
    #    lw=0.15,
    #)
    #ax.plot(
    #    np.quantile(np.sort(synthetic, axis=0), 0.95, axis=1),
    #    np.linspace(0, 1, len(synthetic[:, 0])),
    #    color="k",
    #    alpha=0.25,
    #    lw=0.15,
    #)
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
    elif param == 'cos_tilt' or param == 'chi_eff':
        ax.set_xticks([-1, -0.5, 0, 0.5, 1])
        ax.set_xticklabels([-1, -0.5, 0, 0.5, 1])
        ax.set_xlim(-1, 1)
    elif param == "redshift":
        ax.set_xscale('log')
        ax.set_xlim(0.05, zmax)
    elif param == 'mass_ratio':
        ax.set_xlim(0.05,1)
        ax.legend(frameon=False, fontsize=16)
    else:
        ax.legend(frameon=False, fontsize=16)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_xticklabels([0, 0.25, 0.5, 0.75, 1])
        ax.set_xlim(0, 1)
        
plt.suptitle(f'GWTC-3: B-Spline Model Posterior Predictive Checks', fontsize=22);
fig.tight_layout()
plt.subplots_adjust(wspace=0.05)
plt.savefig(paths.figures / 'ppc_plot.pdf', dpi=300);