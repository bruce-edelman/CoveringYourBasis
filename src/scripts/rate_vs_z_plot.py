import paths
import numpy as np
import matplotlib.pyplot as plt
from utils import plot_mean_and_90CI, read_in_result
import deepdish as dd

def load_plbspline_ppd():
    datadict = dd.io.load(paths.data / 'bsplines_64m1_18q_iid18mag_iid16tilt_pl16z_ppds.h5')
    return datadict['zs'], datadict['Rofz']

def load_o3b_rofz(zs):
    result = read_in_result(paths.data / 'o1o2o3_mass_c_iid_mag_iid_tilt_powerlaw_redshift_result.json')
    lamb = result.posterior['lamb']
    rate = result.posterior['rate']
    dRdzs = [rate[i]*(1+zs)**lamb[i] for i in range(len(lamb))]
    return zs, np.array(dRdzs)

zs, dR = load_plbspline_ppd()
pl_zs, pl_dR = load_o3b_rofz(zs)

def plot_rofz(dRdz, zs, pl_dRdz, pl_zs):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 4))
    me = np.median(dRdz, axis=0)
    ax = plot_mean_and_90CI(ax, zs, dRdz, color='tab:red', bounds=True, label='PL+BSpline', median=True, mean=False)
    ax = plot_mean_and_90CI(ax, pl_zs, pl_dRdz, color='tab:blue', bounds=True, fill_alpha=0.2, label='PL', median=False, mean=False)
    ax.plot(zs, me[0] * (1.0 + zs) ** 2.7, lw=5, alpha=0.15, color="k", label="SFR")
    
    ax.set_xlabel(r"$z$", fontsize=16)
    ax.set_ylabel(r"$\mathcal{R}(z)$", fontsize=16)
    ax.set_yscale("log")
    ax.set_ylim(1,1e3)
    ax.set_xlim(zs[0], 1.5)
    ax.legend()
    return fig

fig = plot_rofz(dR, zs, pl_dR, pl_zs)
plt.title(r'GWTC-3: $\mathcal{R}(z)$', fontsize=18);
fig.tight_layout()
plt.savefig(paths.figures / 'rate_vs_z_plot.pdf', dpi=300);