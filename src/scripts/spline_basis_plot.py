#!/usr/bin/env python

import paths
import numpy as np
import matplotlib.pyplot as plt
from utils import MSpline

ndof = 20
k = 3
xmin = 0
xmax = 1
xlab = 'x'

cmap = 'magma'
figx, figy = 7, 5
fig, axs = plt.subplots(nrows=1, ncols=1, sharey='all', figsize=(figx,figy))

nadd = k - 1
interior_knots = np.linspace(xmin, xmax, ndof+k-2*nadd)
#knots = np.concatenate([np.array([xmin]*nadd), interior_knots, np.array([xmax]*nadd)])
dx = interior_knots[1]-interior_knots[0]
knots = np.concatenate([xmin-np.arange(1,k)[::-1]*dx, interior_knots, xmax+dx*np.arange(1,k)])

coef = np.array([1./ndof]*ndof)
grid = np.linspace(xmin, xmax, 2500)
basis = MSpline(ndof, k=k, knots=knots)(grid)

color_cycle = iter(plt.cm.get_cmap(cmap)(np.linspace(0.3, .9,ndof)[::-1]))
total_color = iter(plt.cm.get_cmap(cmap)([0.05]))


total = 0
for ii in range(ndof):
    basis[ii,0] = 2*basis[ii,1]-basis[ii,2]
    basis[ii,-1] = 2*basis[ii,-2]-basis[ii,-3]
    norm = 1.0 / (sum([np.trapz(basis[i, :], grid) * coef[i] for i in range(ndof)]))
    axs.plot(grid, norm * coef[ii] * basis[ii,:], color=next(color_cycle), lw=4.5)#, label=f'c_{ii}')# = {coef[ii]:0.2f}')
    total += norm * coef[ii] * basis[ii,:]
axs.plot(grid, total, color=next(total_color), lw=6, label=r'$\sum_{i}^{N_\mathrm{dof}} c_{i} M_{i,k}(x)$')
maxy = max(total)

dx = knots[k+3] - knots[k+2]
dy = 1.5*maxy / (ndof)
ys = [dy]*len(knots)
xs = knots.copy()
for i, x in enumerate(knots):
    if np.abs(knots[i-1]-x) < 1e-6:
        ys[i] = ys[i-1] + dy
    if np.abs(xmin-x) < 1e-6:
        xs[i] = knots[i] + 0.1*dx
    if np.abs(xmax-x) < 1e-6:
        xs[i] = knots[i] - 0.1*dx
axs.scatter(xs, ys, s=250, marker='x', color='k', alpha=0.4, label='Knots', lw=2.5, zorder=10)

leg_item = ndof + 2
max_col = 6
ncol = max([leg_item // max_col, 3])
axs.legend(loc='upper center', fancybox=True, shadow=True, ncol=ncol, fontsize=16)
axs.set_xlim(0,1)
axs.set_ylim(0, 1.5)
axs.set_xlabel(xlab, fontsize=18)
axs.grid(True, which="major", ls=":")
axs.set_ylabel(f'p({xlab})', fontsize=18)
axs.tick_params(labelsize=14)
plt.title(f'MSpline Basis: k={k-1}, n={ndof}, from {xlab}=[{xmin}, {xmax}]', fontsize=16)
fig.tight_layout()
plt.savefig(paths.figures / 'spline_basis_plot.pdf', dpi=300)