from argparse import ArgumentParser
from jax import jit
import numpy as np
import jax.numpy as jnp
import numpyro
import pickle
import numpyro.distributions as dist
from tqdm import trange
import matplotlib.pyplot as plt
from gwheiro.models.msplines.seperable import MSplinePrimaryPowerlawRatio
from gwheiro.models.gwpopulation import PowerlawRedshiftModel

def load_base_parser():
    parser = ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='/home/bruce.edelman/projects/GWTC3_allevents/')
    parser.add_argument('--inj-file', type=str, default='/home/bruce.edelman/projects/GWTC3_allevents/o1o2o3_mixture_injections.hdf5')
    parser.add_argument('--mmin', type=float, default=3.0)
    parser.add_argument('--mmax', type=float, default=100.0)
    parser.add_argument('--mass-knots', type=int, default=50)
    parser.add_argument('--jax-gpu-device', type=int, default=0)
    parser.add_argument('--spin-knots', type=int, default=16)
    parser.add_argument('--chains', type=int, default=4)
    parser.add_argument('--samples', type=int, default=2000)
    parser.add_argument('--warmup', type=int, default=250)
    parser.add_argument('--dirichlet', action='store_true', default=False)
    parser.add_argument('--sample-prior', action='store_true', default=False)
    parser.add_argument('--chieff-knots', type=int, default=24)
    parser.add_argument('--chip-knots', type=int, default=12)
    return parser


def setup_mass_MSpline_model(injdata, pedata, pmap, nknots, mmax=100.0, k=4):
    m1per_pe = np.min(np.min(pedata[pmap['mass_1']], axis=0))
    m1per_inj = np.min(injdata[pmap['mass_1']])
    mmin = max([m1per_inj, m1per_pe])
    print(f"Setting up MSpline primary mass model with min m1={mmin}")
    print(f"Min m2 mass is: {np.min(np.min(pedata[pmap['mass_ratio']]*pedata[pmap['mass_1']], axis=0))}")
    interior_logknots = np.linspace(np.log10(mmin), np.log10(mmax), nknots-k+2)
    dx = interior_logknots[1] - interior_logknots[0]
    logknots = np.concatenate([np.log10(mmin)-dx*np.arange(1,k)[::-1], interior_logknots, np.log10(mmax)+dx*np.arange(1,k)])
    knots = 10**(logknots)
    model = MSplinePrimaryPowerlawRatio(nknots, pedata[pmap['mass_1']],injdata[pmap['mass_1']],
                                       knots=knots)
    return model, mmin


def setup_redshift_model(injdata, pedata, pmap):
    z_pe = pedata[pmap['redshift']]
    z_inj = injdata[pmap['redshift']]
    model = PowerlawRedshiftModel(z_pe, z_inj)
    return model


def calculate_penalty(coefs, inv_var, Lambda=None, degree=1):
    if Lambda is None:
        Lambda=jnp.ones(len(coefs)-degree)
    D = jnp.diff(jnp.identity(len(coefs)), n=degree)
    delta_c = jnp.dot(coefs,D)
    return -0.5 * inv_var * jnp.sum(delta_c*Lambda*delta_c.T)


def get_adaptive_Lambda(label, nknots, degree, omega=0.5):
    lam = numpyro.sample(f"lambda_{label}", dist.Gamma(omega, omega), sample_shape=(nknots-degree-1,))
    l = [1.0]
    for i,la in enumerate(lam):
        l.append(l[i]*la)
    return jnp.diag(jnp.array(l))


def mixture_smoothing_parameter(label, n_mix=20, log10bmin=-5, log10bmax=5):
    bs = jnp.logspace(log10bmin, log10bmax, num=n_mix)
    ps = numpyro.sample(f"{label}_ps", dist.Dirichlet(jnp.ones(n_mix)))
    gs = numpyro.sample(f"{label}_gs", dist.Gamma(jnp.ones_like(bs), bs))
    return jnp.sum(ps*gs)


def calculate_m1q_ppds(mcoefs, rate, betas, mass_model, nknots, mmin, m1mmin, mmax, k=4):
    interior_logknots = np.linspace(np.log10(m1mmin), np.log10(mmax), nknots-k+2)
    dx = interior_logknots[1] - interior_logknots[0]
    logknots = np.concatenate([np.log10(m1mmin)-dx*np.arange(1,k)[::-1], interior_logknots, np.log10(mmax)+dx*np.arange(1,k)])
    knots = 10**(logknots)
    ms = jnp.linspace(mmin, mmax, 1000)
    qs = jnp.linspace(mmin/mmax, 1, 1000)
    mm, qq = jnp.meshgrid(ms, qs)
    mass_pdf = mass_model(nknots, mm, ms, knots=knots)
    mpdfs = []
    qpdfs = []
    
    @jit
    def calc_pdf(mcs, bet, r):
        p_mq = mass_pdf(mm, qq, bet, mmin, mcs)
        p_mq = jnp.where(jnp.isinf(p_mq) | jnp.isnan(p_mq), 0, p_mq)
        p_mq = jnp.where(jnp.greater(mm, mmax) | jnp.less(mm, mmin) | jnp.less(mm*qq, mmin), 0, p_mq)
        p_m = jnp.trapz(p_mq, qs, axis=0)
        p_q = jnp.trapz(p_mq, ms, axis=1)
        return r * p_m, r * p_q
    # loop through hyperposterior samples
    for ii in trange(mcoefs.shape[0]):
        p_m, p_q = calc_pdf(mcoefs[ii], betas[ii], rate[ii])
        mpdfs.append(p_m)
        qpdfs.append(p_q)
    return np.array(mpdfs), np.array(qpdfs), ms, qs

def calculate_ind_spin_ppds(pcoefs, scoefs, rate, model, nknots, xmin, xmax, k=4):
    interior_knots = np.linspace(xmin,xmax,nknots-k+2)
    dx = interior_knots[1]-interior_knots[0]
    knots = np.concatenate([xmin-dx*np.arange(1,k)[::-1], interior_knots,xmax+dx*np.arange(1,k)])
    xs = jnp.linspace(xmin, xmax, 1000)
    pdf = model(nknots, nknots, xs, xs, xs, xs, knots1=knots, knots2=knots, order1=k-1, order2=k-1)
    ppdfs = []
    spdfs = []
    @jit
    def calc_pdf(pcs, scs, r):
        p_p = pdf.primary_model(1, pcs) * r
        p_s = pdf.secondary_model(1, scs) * r
        return p_p, p_s
    # loop through hyperposterior samples
    for ii in trange(pcoefs.shape[0]):
        pp, ps = calc_pdf(pcoefs[ii], scoefs[ii], rate[ii])
        ppdfs.append(pp)
        spdfs.append(ps)
    return jnp.array(ppdfs), jnp.array(spdfs), xs


def calculate_rate_of_z_ppds(lamb, rate, model):
    zs = model.zs
    rs = []
    for ii in range(lamb.shape[0]):
        rs.append(rate[ii]*jnp.power(1.0+zs, lamb[ii]))
    return jnp.array(rs), zs


def plot_mass_dist(pm1s, pqs, ms, qs, mmin=5.0, mmax=100.0):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(14,4))
    for ax, xs, ps, lab in zip(axs, [ms,qs], [pm1s,pqs], ['m1', 'q']):
        me = np.mean(ps, axis=0)
        low = np.percentile(ps, 5, axis=0)
        high = np.percentile(ps, 95, axis=0)
        if lab == 'm1':
            ax.fill_between(xs, low, high, color='tab:blue', alpha=0.15)
            ax.plot(xs, me, color='tab:blue', lw=4, alpha=0.5, label='MSpline')
        else:
            ax.fill_between(xs, low, high, color='tab:blue', alpha=0.15)
            ax.plot(xs, me, color='tab:blue', lw=4, alpha=0.5, label='MSpline')
    axs[0].set_xlabel(r'$m_1 \,\,[M_\odot]$', fontsize=16)
    axs[1].set_xlabel(r'$q$', fontsize=16)
    axs[0].set_ylabel(r'$\frac{d\mathcal{R}}{dm_1}$', fontsize=16)
    axs[1].set_ylabel(r'$\frac{d\mathcal{R}}{dq}$', fontsize=16)
    axs[0].set_yscale('log')
    axs[1].set_yscale('log')
    axs[0].set_xscale('log')
    axs[0].set_xlim(mmin, mmax)
    axs[0].set_ylim(2e-3, 1e1)
    axs[1].set_xlim(3.0/mmax, 1)
    axs[1].set_ylim(1e-1,1e2)
    return fig

def plot_ind_spin_dist(ppmags, psmags, pptilts, pstilts, mags, tilts):
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(14,9), sharey='row')

    ps = ppmags
    ax = axs[0,0]
    me = np.mean(ps, axis=0)
    low = np.percentile(ps, 5, axis=0)
    high = np.percentile(ps, 95, axis=0)
    ax.fill_between(mags, low, high, color='tab:blue', alpha=0.15)
    ax.plot(mags, me, color='tab:blue', lw=4, alpha=0.5, label='MSpline')
    
    ps = psmags
    ax = axs[0,1]
    me = np.mean(ps, axis=0)
    low = np.percentile(ps, 5, axis=0)
    high = np.percentile(ps, 95, axis=0)
    ax.fill_between(mags, low, high, color='tab:blue', alpha=0.15)
    ax.plot(mags, me, color='tab:blue', lw=4, alpha=0.5, label='MSpline')
    
    ps = pptilts
    ax = axs[1,0]
    me = np.mean(ps, axis=0)
    low = np.percentile(ps, 5, axis=0)
    high = np.percentile(ps, 95, axis=0)
    ax.fill_between(tilts, low, high, color='tab:blue', alpha=0.15)
    ax.plot(tilts, me, color='tab:blue', lw=4, alpha=0.5, label='MSpline')
    
    ps = pstilts
    ax = axs[1,1]
    me = np.mean(ps, axis=0)
    low = np.percentile(ps, 5, axis=0)
    high = np.percentile(ps, 95, axis=0)
    ax.fill_between(tilts, low, high, color='tab:blue', alpha=0.15)
    ax.plot(tilts, me, color='tab:blue', lw=4, alpha=0.5, label='MSpline')

    axs[0,0].set_xlabel(r'$a_1$', fontsize=16)
    axs[1,0].set_xlabel(r'$\cos(\theta_1)$', fontsize=16)
    axs[0,0].set_ylabel(r'$\frac{d\mathcal{R}}{da_1}$', fontsize=16)
    axs[1,0].set_ylabel(r'$\frac{d\mathcal{R}}{d\cos(\theta_1)}$', fontsize=16)
    axs[0,0].set_xlim(0,1)
    axs[0,0].set_ylim(0)
    axs[1,0].set_xlim(-1,1)
    axs[1,0].set_ylim(0)
    axs[0,1].set_xlabel(r'$a_2$', fontsize=16)
    axs[1,1].set_xlabel(r'$\cos(\theta_2)$', fontsize=16)
    axs[0,1].set_ylabel(r'$\frac{d\mathcal{R}}{da_2}$', fontsize=16)
    axs[1,1].set_ylabel(r'$\frac{d\mathcal{R}}{d\cos(\theta_2)}$', fontsize=16)
    axs[0,1].set_xlim(0,1)
    axs[0,1].set_ylim(0)
    axs[1,1].set_xlim(-1,1)
    axs[1,1].set_ylim(0)
    return fig


def calculate_iid_spin_ppds(coefs, rate, model, nknots, xmin, xmax, k=4):
    interior_knots = np.linspace(xmin,xmax,nknots-k+2)
    dx = interior_knots[1]-interior_knots[0]
    knots = np.concatenate([xmin-dx*np.arange(1,k)[::-1], interior_knots,xmax+dx*np.arange(1,k)])
    xs = jnp.linspace(xmin, xmax, 1000)
    pdf = model(nknots, xs, xs, xs, xs, knots=knots, order=k-1)
    pdfs = []
    @jit
    def calc_pdf(cs, r):
        return pdf.primary_model(1, cs) * r
    # loop through hyperposterior samples
    for ii in trange(coefs.shape[0]):
        p = calc_pdf(coefs[ii], rate[ii])
        pdfs.append(p)
    return jnp.array(pdfs), xs


def plot_iid_spin_dist(pmags, ptilts, mags, tilts):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(14,4))
    for ax, xs, ps, lab in zip(axs, [mags,tilts], [pmags,ptilts], ['a', 'cos(theta)']):
        me = np.mean(ps, axis=0)
        low = np.percentile(ps, 5, axis=0)
        high = np.percentile(ps, 95, axis=0)
        ax.fill_between(xs, low, high, color='tab:blue', alpha=0.15)
        ax.plot(xs, me, color='tab:blue', lw=4, alpha=0.5, label='MSpline')

    axs[0].set_xlabel(r'$a$', fontsize=16)
    axs[1].set_xlabel(r'$\cos(\theta)$', fontsize=16)
    axs[0].set_ylabel(r'$\frac{d\mathcal{R}}{da}$', fontsize=16)
    axs[1].set_ylabel(r'$\frac{d\mathcal{R}}{d\cos(\theta)}$', fontsize=16)
    axs[0].set_xlim(0,1)
    axs[0].set_ylim(0)
    axs[1].set_xlim(-1,1)
    axs[1].set_ylim(0)
    return fig


def calculate_chi_ppds(coefs, rate, model, nknots, xmin, xmax, k=4, knots=None):
    if knots is None:
        interior_knots = np.linspace(xmin,xmax,nknots-k+2)
        dx = interior_knots[1]-interior_knots[0]
        knots = np.concatenate([xmin-dx*np.arange(1,k)[::-1], interior_knots,xmax+dx*np.arange(1,k)])
    xs = jnp.linspace(xmin, xmax, 1000)
    pdf = model(nknots, xs, xs, knots=knots, order=k-1)
    pdfs = []
    @jit
    def calc_pdf(cs, r):
        return pdf(1, cs) * r
    # loop through hyperposterior samples
    print("calculating chi eff ppds...")

    for ii in trange(coefs.shape[0]):
        p = calc_pdf(coefs[ii], rate[ii])
        pdfs.append(p)
    return jnp.array(pdfs), xs

def plot_chieff_dist(ps,xs):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7,4))
    me = np.mean(ps, axis=0)
    low = np.percentile(ps, 5, axis=0)
    high = np.percentile(ps, 95, axis=0)
    ax.fill_between(xs, low, high, color='tab:blue', alpha=0.15)
    ax.plot(xs, me, color='tab:blue', lw=4, alpha=0.5, label='MSpline')
    ax.set_xlabel(r'$\chi_\mathrm{eff}$', fontsize=16)
    ax.set_ylabel(r'$\frac{d\mathcal{R}}{d\chi_\mathrm{eff}}$', fontsize=16)
    ax.set_xlim(-1,1)
    ax.set_ylim(0)
    return fig


def plot_chip_dist(ps,xs):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7,4))
    me = np.mean(ps, axis=0)
    low = np.percentile(ps, 5, axis=0)
    high = np.percentile(ps, 95, axis=0)
    ax.fill_between(xs, low, high, color='tab:blue', alpha=0.15)
    ax.plot(xs, me, color='tab:blue', lw=4, alpha=0.5, label='MSpline')
    ax.set_xlabel(r'$\chi_\mathrm{p}$', fontsize=16)
    ax.set_ylabel(r'$\frac{d\mathcal{R}}{d\chi_\mathrm{p}}$', fontsize=16)
    ax.set_xlim(0,1)
    ax.set_ylim(0)
    return fig

def load_in_toms_pe(nsamps=3500):
    params = ['mass_1', 'mass_2', 'redshift', 'chi_eff', 'chi_p', 'prior'] # 
    with open('sampleDict_FAR_1_in_1_yr_11-29.pickle', 'rb') as f:
        pedict = pickle.load(f)
        #pedict = lalprior(pedict)
        ct = 0
        nsamps = min([len(pedict[e]['z']) for e in pedict.keys()])
        print(f"Saving {nsamps} samples from each event...")
        samps = np.empty((len(params),69,nsamps))
        for e in pedict.keys():
            if e == 'S190814bv':
                continue
            post = pedict[e]
            m1 = post['m1']
            q = post['m2'] / post['m1']
            z = post['z']
            chieff = post['Xeff']
            chip = post['Xp']
            prior = post['m1'] * post['joint_priors'] * (1.0 + z)**(1.7) / post['weights']
            idxs = np.random.choice(len(m1), size=nsamps, replace=False)
            samps[0,ct,:] = m1[idxs]
            samps[1,ct,:] = q[idxs]
            samps[2,ct,:] = z[idxs]
            samps[3,ct,:] = chieff[idxs]
            samps[4,ct,:] = chip[idxs]
            samps[5,ct,:] = prior[idxs]
            ct += 1
    return samps

def load_in_toms_injs():
    params = ['mass_1', 'mass_2', 'redshift', 'chi_eff', 'chi_p', 'prior'] #'chi_p', 
    with open('injectionDict_10-20_directMixture_FAR_1_in_1.pickle', 'rb') as f:
        injdict = pickle.load(f)
        m1 = injdict['m1']
        q = injdict['m2'] / injdict['m1']
        z = injdict['z']
        chieff = injdict['Xeff']
        chip = injdict['Xp']
        prior = injdict['m1'] * 1.0 / injdict['weights']
    injs = np.array([m1, q, z, chieff, chip, prior])# 
    return injs