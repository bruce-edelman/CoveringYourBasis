import paths
import numpy as np
import matplotlib.pyplot as plt
import deepdish as dd

fig, ax = plt.subplots(1,1,figsize=(7,5))
ppds = dd.io.load(paths.data / "chi_eff_ppds.h5")
default = ppds["Default"]
mspl = ppds["MSplineInd"]
plt.plot(
    default["chieffs"],
    np.median(default["pchieff"], axis=0),
    lw=2,
    color="tab:blue",
    label="Default",
)
plt.fill_between(
    default["chieffs"],
    *np.percentile(default["pchieff"], (5, 95), axis=0),
    color="tab:blue",
    alpha=0.15,
)
plt.plot(
    mspl["chieffs"],
    np.median(mspl["pchieff"], axis=0),
    lw=2,
    color="tab:purple",
    label="MSpline Ind Spins",
)
plt.fill_between(
    mspl["chieffs"],
    *np.percentile(mspl["pchieff"], (5, 95), axis=0),
    color="tab:purple",
    alpha=0.15,
)
handpicked_ppds = dd.io.load(
    paths.data / "mspline_50m1_24chieff_smoothprior_powerlaw_q_z_fitlamb_ppds.h5"
)
xs = handpicked_ppds["chieffs"]
pchiefss = handpicked_ppds["dRdchieff"]
for i in range(1500):
    norm = np.trapz(pchiefss[i, :], x=xs)
    pchiefss[i, :] /= norm
plt.plot(
    xs,
    np.median(pchiefss, axis=0),
    lw=2,
    color="tab:red",
    label="MSpline Chi Effective",
)
plt.fill_between(
    xs, *np.percentile(pchiefss, (5, 95), axis=0), color="tab:red", alpha=0.15
)
plt.xlim(-0.75, 0.5)
plt.ylim(0, 6.5)
plt.legend(frameon=False, fontsize=14)
ax.grid(True, which="major", ls=":")
ax.tick_params(labelsize=14)
plt.xlabel(r"$\chi_\mathrm{eff}$", fontsize=18)
plt.ylabel(r"$p(\chi_\mathrm{eff})$", fontsize=18)
plt.title(f"GWTC-3: Effective Spin Distribution", fontsize=18)
plt.savefig(paths.figures / "chi_eff.pdf", dpi=300)
