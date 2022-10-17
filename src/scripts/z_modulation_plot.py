import paths
import numpy as np
import matplotlib.pyplot as plt
from utils import plot_mean_and_90CI
import deepdish as dd

def load_plbspline_ppd():
    datadict = dd.io.load(paths.data / 'bsplines_64m1_18q_iid18mag_iid18tilt_pl18z_ppds.h5')
    prior_datadict = dd.io.load(paths.data / 'bsplines_64m1_18q_iid18mag_iid18tilt_pl18z_prior_ppds.h5')
    posterior = dd.io.load(paths.data / 'bsplines_64m1_18q_iid18mag_iid18tilt_pl18z_posterior_samples.h5')
    prior = dd.io.load(paths.data / 'bsplines_64m1_18q_iid18mag_iid18tilt_pl18z_prior_samples.h5')
    return datadict['zs'], datadict['Rofz'], prior_datadict['zs'], prior_datadict['Rofz'], posterior['lamb'], prior['lamb'], posterior['rate']

def get_modulation(R,z,L,rate=None):
    N = len(L)
    mod = np.zeros_like(R)
    for i in range(N):
        if rate is not None:
            rate_of_z = R[i,:] / rate[i]
        else:
            rate_of_z = R[i,:]
        mod[i,:] = np.log(rate_of_z / (1+z)**(L[i]))
    return mod

zs, Rofz, pr_zs, pr_Rofz, lamb, pr_lamb, rate = load_plbspline_ppd()
modulation = get_modulation(Rofz, zs, lamb, rate=rate)
prior_modulation = get_modulation(pr_Rofz, pr_zs, pr_lamb)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))
ax = plot_mean_and_90CI(ax, zs, modulation, color='tab:red', label = 'This Work', bounds=True, median=True, mean=False)

ax.plot(pr_zs, np.percentile(prior_modulation,5,axis=0), lw=2, ls='--', color="k", label="Prior")
ax.plot(pr_zs, np.percentile(prior_modulation,95,axis=0), lw=2, ls='--', color="k")
ax.axhline(0, color="k", lw=2)

ax.set_xlabel(r"$z$", fontsize=18)
ax.set_ylabel(r"$B(z)$", fontsize=18)
ax.set_xscale('log')
ax.set_ylim(-1.2,1.2)
ax.set_xlim(zs[0], zs[-1])
ax.grid(True, which="major", ls=":")
plt.title(r'GWTC-3: Redshift Modulation', fontsize=18);
fig.tight_layout()
plt.savefig(paths.figures / 'z_modulation_plot.pdf', dpi=300);