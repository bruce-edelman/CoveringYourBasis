#!/usr/bin/env python

import paths
import numpy as np
import matplotlib.pyplot as plt
import deepdish as dd


params = [
    "a",
    "cos_tilt",
    "chi_eff",
    "chi_p",
]

param_latex = {
    "a": r'$a$',
    "cos_tilt": r'$\cos{\theta}$',
    "chi_eff": r'$\chi_\mathrm{eff}$',
    "chi_p": r'$\chi_\mathrm{p}$',
}

def chi_eff(q,a1,a2,ct1,ct2):
    return (a1*ct1 + q*a2*ct2) / (1.0 + q)

def chi_p(q,a1,a2,ct1,ct2):
    st1 = np.sqrt(1.0 - ct1**2)
    st2 = np.sqrt(1.0 - ct2**2)
    return np.maximum(a1*st1, (3+4*q)/(4+3*q)*q*a2*st2)

nplot = len(params)
figx, figy = 16,5
fig, axs = plt.subplots(nrows=1, ncols=nplot, sharey='row', figsize=(figx,figy))

Nobs = 69
po = dd.io.load(paths.data / 'bsplines_64m1_18q_iid18mag_iid18tilt_pl18z_posterior_samples.h5')

obsdata = {'a1': np.array([po[f"a_1_obs_event_{i}"] for i in range(Nobs)]), 
           'a2': np.array([po[f"a_2_obs_event_{i}"] for i in range(Nobs)]), 
           'ct1': np.array([po[f"cos_tilt_1_obs_event_{i}"] for i in range(Nobs)]), 
           'ct2': np.array([po[f"cos_tilt_2_obs_event_{i}"] for i in range(Nobs)]), 
           'q': np.array([po[f"mass_ratio_obs_event_{i}"] for i in range(Nobs)])}
synthdata = {'a1': np.array([po[f"a_1_pred_event_{i}"] for i in range(Nobs)]), 
           'a2': np.array([po[f"a_2_pred_event_{i}"] for i in range(Nobs)]), 
           'ct1': np.array([po[f"cos_tilt_1_pred_event_{i}"] for i in range(Nobs)]), 
           'ct2': np.array([po[f"cos_tilt_2_pred_event_{i}"] for i in range(Nobs)]), 
           'q': np.array([po[f"mass_ratio_pred_event_{i}"] for i in range(Nobs)])}

for ax, param in zip(axs, params):
    if param == 'a':
        observed1 = obsdata['a1']
        observed2 = obsdata['a2']
        synthetic1 = synthdata['a1']
        synthetic2 = synthdata['a2']
        observed = np.concatenate([observed1, observed2]).reshape((Nobs,2*observed1.shape[1]))
        synthetic = np.concatenate([synthetic1, synthetic2]).reshape((Nobs,2*synthetic1.shape[1]))
    elif param == 'cos_tilt':
        observed1 = obsdata['ct1']
        observed2 = obsdata['ct2']
        synthetic1 = synthdata['ct1']
        synthetic2 = synthdata['ct2']
        observed = np.concatenate([observed1, observed2]).reshape((Nobs,2*observed1.shape[1]))
        synthetic = np.concatenate([synthetic1, synthetic2]).reshape((Nobs,2*synthetic1.shape[1]))
    elif param == 'chi_eff':
        observed = chi_eff(obsdata['q'], obsdata['a1'],obsdata['a2'],obsdata['ct1'],obsdata['ct2'])
        synthetic = chi_eff(synthdata['q'], synthdata['a1'], synthdata['a2'], synthdata['ct1'], synthdata['ct2'])
    else:
        observed = chi_p(obsdata['q'], obsdata['a1'],obsdata['a2'],obsdata['ct1'],obsdata['ct2'])
        synthetic = chi_p(synthdata['q'], synthdata['a1'], synthdata['a2'], synthdata['ct1'], synthdata['ct2'])

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
    if param == 'cos_tilt':
        ax.set_xticks([-1, -0.5, 0, 0.5, 1])
        ax.set_xticklabels([-1, -0.5, 0, 0.5, 1])
        ax.set_xlim(-1, 1)
    elif param == 'chi_eff':
        ax.set_xticks([-0.5, -0.25, 0, 0.25, 0.5, 1])
        ax.set_xticklabels([-0.5, -0.25, 0, 0.25, 0.5, 1])
        ax.set_xlim(-0.5, 1)
    else:
        ax.legend(frameon=False, fontsize=16)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_xticklabels([0, 0.25, 0.5, 0.75, 1])
        ax.set_xlim(0, 1)
        
plt.suptitle(f'GWTC-3: B-Spline Spin Model Posterior Predictive Checks', fontsize=22);
fig.tight_layout()
plt.subplots_adjust(wspace=0.05)
plt.savefig(paths.figures / 'spin_ppc_plot.pdf', dpi=300);