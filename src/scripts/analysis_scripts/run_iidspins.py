import arviz as az
import deepdish as dd
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpyro
import numpy as np
from tqdm import trange
from jax import jit, random
from numpyro import distributions as dist
from numpyro.infer import MCMC, NUTS
from argparse import ArgumentParser
from gwinferno.models.spline_perturbation import PowerlawSplineRedshiftModel
from gwinferno.interpolation import LogXLogYBSpline, LogYBSpline, LogXBSpline
from gwinferno.models.bsplines.separable import BSplinePrimaryBSplineRatio, BSplineIIDSpinMagnitudes, BSplineIIDSpinTilts
from gwinferno.models.bsplines.smoothing import calculate_penalty
from gwinferno.data_collection import load_injections, load_posterior_samples
from gwinferno.analysis import hierarchical_likelihood
from gwinferno.plotting import plot_mass_dist, plot_iid_spin_dist, plot_m1_vs_z_ppc, plot_rofz, plot_ppc_brontosaurus
az.style.use("arviz-darkgrid")


def load_parser():
    parser = ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='/home/bruce.edelman/projects/GWTC3_allevents/')
    parser.add_argument('--inj-file', type=str, default='/home/bruce.edelman/projects/GWTC3_allevents/o1o2o3_mixture_injections.hdf5')
    parser.add_argument('--outdir', type=str, default='results')
    parser.add_argument('--mmin', type=float, default=3.0)
    parser.add_argument('--mmax', type=float, default=100.0)
    parser.add_argument('--mass-knots', type=int, default=100)
    parser.add_argument('--mag-knots', type=int, default=30)
    parser.add_argument('--q-knots', type=int, default=30)
    parser.add_argument('--tilt-knots', type=int, default=25)
    parser.add_argument('--z-knots', type=int, default=20)
    parser.add_argument('--chains', type=int, default=1)
    parser.add_argument('--samples', type=int, default=1500)
    parser.add_argument('--thinning', type=int, default=1)
    parser.add_argument('--warmup', type=int, default=500)
    parser.add_argument('--skip-inference', action='store_true', default=False)
    return parser.parse_args()


def setup_mass_BSpline_model(injdata, pedata, pmap, nknots, qknots, mmin=3.0, mmax=100.0):
    m1per_pe = np.min(np.min(pedata[pmap['mass_1']], axis=0))
    m1per_inj = np.min(injdata[pmap['mass_1']])
    m1min = max([m1per_inj, m1per_pe])
    m1min = 6.5
    print(f"Basis Spline model in m1 w/ {nknots} knots logspaced from {m1min} to {mmax}...")
    print(f"Basis Spline model in q w/ {qknots} knots linspaced from {mmin/mmax} to 1...")

    model = BSplinePrimaryBSplineRatio(
        nknots,
        qknots,
        pedata[pmap['mass_1']],
        injdata[pmap['mass_1']],
        pedata[pmap['mass_ratio']],
        injdata[pmap['mass_ratio']],
        m1min=m1min,
        m2min=mmin,
        mmax=mmax,
        basis_m=LogXLogYBSpline,
        basis_q=LogYBSpline,
        )
    return model, m1min


def setup_spin_BSpline_model(injdata, pedata, pmap, magnknots, tiltnknots):
    magmodel = BSplineIIDSpinMagnitudes(magnknots, pedata[pmap['a_1']], pedata[pmap['a_2']], 
                                            injdata[pmap['a_1']], injdata[pmap['a_2']], basis=LogYBSpline, normalize=True)
    tiltmodel = BSplineIIDSpinTilts(tiltnknots, pedata[pmap['cos_tilt_1']], pedata[pmap['cos_tilt_2']], 
                                        injdata[pmap['cos_tilt_1']], injdata[pmap['cos_tilt_2']], basis=LogYBSpline, normalize=True)
        
    return {'mag': magmodel, 'tilt': tiltmodel}


def setup_redshift_model(z_knots, injdata, pedata, pmap):
    z_pe = pedata[pmap['redshift']]
    z_inj = injdata[pmap['redshift']]
    model = PowerlawSplineRedshiftModel(z_knots, z_pe, z_inj, basis=LogXBSpline)
    return model


def setup(args):
    injections = load_injections(args.inj_file, spin=True)
    pe_samples, names = load_posterior_samples(args.data_dir, spin=True)
    param_names = ['mass_1', 'mass_ratio', 'redshift', 'a_1', 'a_2', 'cos_tilt_1', 'cos_tilt_2', 'prior']
    param_map = {p: i for i, p in enumerate(param_names)}
    injdata = jnp.array([injections[k] for k in param_names])
    pedata = jnp.array([[pe_samples[e][p] for e in names] for p in param_names])
    nObs = pedata.shape[1]
    total_inj = injections["total_generated"]
    obs_time = injections["analysis_time"]
    
    mass_model, min_m1_interp = setup_mass_BSpline_model(injdata, pedata, param_map, args.mass_knots,
                                                         args.q_knots, mmin=args.mmin, mmax=args.mmax)
    z_model = setup_redshift_model(args.z_knots, injdata, pedata, param_map)
    spin_models = setup_spin_BSpline_model(injdata, pedata, param_map, args.mag_knots, args.tilt_knots)
    injdict = {k: injdata[param_map[k]] for k in param_names}
    pedict = {k: pedata[param_map[k]] for k in param_names}
    
    print(f"{len(injdict['redshift'])} found injections out of {total_inj} total")
    print(f"Observed {nObs} events, each with {pedict['redshift'].shape[1]} samples, over an observing time of {obs_time} yrs")    
    
    return mass_model, spin_models, z_model, pedict, injdict, total_inj, nObs, obs_time, min_m1_interp


def model(mass_model, spin_models, z_model, pedict, injdict, total_inj, Nobs, Tobs, sample_prior=False):
    mass_knots = mass_model.primary_model.nknots
    q_knots = mass_model.ratio_model.nknots
    mag_model = spin_models['mag']
    tilt_model = spin_models['tilt']
    mag_knots = mag_model.primary_model.nknots 
    tilt_knots = tilt_model.primary_model.nknots
    z_knots = z_model.nknots 
    
    mass_cs = numpyro.sample('mass_cs', dist.Normal(0,5), sample_shape=(mass_knots,))
    mass_tau = numpyro.sample("mass_tau", dist.Uniform(1,100))
    numpyro.factor("mass_log_smoothing_penalty", calculate_penalty(mass_cs, mass_tau, degree=2))

    q_cs = numpyro.sample('q_cs', dist.Normal(0,4), sample_shape=(q_knots,))
    q_tau = numpyro.sample("q_tau", dist.Uniform(1,10))
    numpyro.factor("q_log_smoothing_penalty", calculate_penalty(q_cs, q_tau, degree=2))

    mag_cs = numpyro.sample('mag_cs', dist.Normal(0,2), sample_shape=(mag_knots,))
    mag_tau = numpyro.sample("mag_tau", dist.Uniform(1,10))
    numpyro.factor("mag_log_smoothing_penalty", calculate_penalty(mag_cs, mag_tau, degree=2))
    tilt_cs = numpyro.sample('tilt_cs', dist.Normal(0,2), sample_shape=(tilt_knots,))
    tilt_tau = numpyro.sample("tilt_tau", dist.Uniform(1,10))
    numpyro.factor("tilt_log_smoothing_penalty", calculate_penalty(tilt_cs, tilt_tau, degree=2))

    lamb = numpyro.sample("lamb", dist.Normal(0,3))
    z_cs = numpyro.sample('z_cs', dist.Normal(0,1), sample_shape=(z_knots,))
    z_tau = numpyro.sample("z_tau",dist.Uniform(1,10))
    numpyro.factor("z_log_smoothing_penalty", calculate_penalty(z_cs, z_tau, degree=2))
    
    if not sample_prior:
        def get_weights(z,prior):
            p_m1q = mass_model(len(z.shape), mass_cs, q_cs)
            p_a1a2 = mag_model(len(z.shape), mag_cs)
            p_ct1ct2 = tilt_model(len(z.shape), tilt_cs)
            p_z = z_model(z, lamb, z_cs)
            wts = p_m1q*p_a1a2*p_ct1ct2*p_z/prior
            return jnp.where(jnp.isnan(wts) | jnp.isinf(wts), 0, wts)
        peweights = get_weights(pedict['redshift'], pedict['prior'])
        injweights = get_weights(injdict['redshift'], injdict['prior'])
        hierarchical_likelihood(peweights, injweights, total_inj=total_inj, Nobs=Nobs, Tobs=Tobs, 
                                surv_hypervolume_fct=z_model.normalization, vtfct_kwargs=dict(lamb=lamb,cs=z_cs), marginalize_selection=False,
                                min_neff_cut=True, posterior_predictive_check=True, pedata=pedict, injdata=injdict, 
                                param_names=['mass_1', 'mass_ratio', 'a_1', 'a_2', 'cos_tilt_1', 'cos_tilt_2', 'redshift'])


def calculate_m1q_ppds(mcoefs, qcoefs, mass_model, nknots, qknots, mmin, m1mmin, mmax): 
    ms = np.linspace(m1mmin, mmax, 1000)
    qs = np.linspace(mmin/mmax, 1, 750)
    mm, qq = np.meshgrid(ms, qs)
    mass_pdf = mass_model(nknots, qknots, mm, ms, qq, qs, m1min=m1mmin, m2min=mmin, mmax=mmax, 
                          basis_m=LogXLogYBSpline, basis_q=LogYBSpline, normalize=False)
    mpdfs = np.zeros((mcoefs.shape[0], len(ms)))
    qpdfs = np.zeros((qcoefs.shape[0], len(qs)))
    
    def calc_pdf(mcs, qcs):#, r):
        p_mq = mass_pdf(2, mcs, qcs) 
        p_mq = jnp.where(jnp.less(mm, m1mmin) | jnp.less(mm*qq, mmin), 0, p_mq)
        p_m = jnp.trapz(p_mq, qs, axis=0)
        p_q = jnp.trapz(p_mq, ms, axis=1)
        return p_m / jnp.trapz(p_m, ms), p_q / jnp.trapz(p_q, qs)
    calc_pdf = jit(calc_pdf)
    _ = calc_pdf(mcoefs[0], qcoefs[0])
    # loop through hyperposterior samples
    for ii in trange(mcoefs.shape[0]):
        mpdfs[ii], qpdfs[ii] = calc_pdf(mcoefs[ii], qcoefs[ii])#, rate[ii])
    return mpdfs, qpdfs, ms, qs

def calculate_iid_spin_ppds(coefs, model, nknots, xmin, xmax, k=4):
    xs = np.linspace(xmin, xmax, 750)
    pdf = model(nknots, xs, xs, xs, xs, order=k-1, basis=LogYBSpline, normalize=True)
    pdfs = np.zeros((coefs.shape[0], len(xs)))

    def calc_pdf(cs):#, r):
        return pdf.primary_model(1, cs)# * r
    calc_pdf = jit(calc_pdf)
    _ = calc_pdf(coefs[0])
    # loop through hyperposterior samples
    for ii in trange(coefs.shape[0]):
        pdfs[ii] = calc_pdf(coefs[ii])
    return pdfs, xs

def calculate_rate_of_z_ppds(lamb, z_cs, rate, model):
    zs = model.zs
    rs = np.zeros((len(lamb), len(zs)))
    def calc_rz(cs,l,r):
        return r * jnp.power(1.0 + zs, l) * jnp.exp(model.interpolator.project(model.norm_design_matrix, (model.nknots, 1), cs))
    calc_rz = jit(calc_rz)
    _ = calc_rz(z_cs[0],lamb[0],rate[0])
    for ii in trange(lamb.shape[0]):
        rs[ii] = calc_rz(z_cs[ii], lamb[ii], rate[ii])
    return rs, zs

    
def main():
    args = load_parser()
    label=f'{args.outdir}/bsplines_{args.mass_knots}m1_{args.q_knots}q_iid{args.mag_knots}mag_iid{args.tilt_knots}tilt_pl{args.z_knots}z'
    mass, spin, z, pedict, injdict, total_inj, nObs, obs_time, min_m1 = setup(args)
    if not args.skip_inference:
        RNG = random.PRNGKey(0)
        MCMC_RNG, PRIOR_RNG, _RNG = random.split(RNG, num=3)
        kernel = NUTS(model)
        mcmc = MCMC(kernel, thinning=args.thinning, num_warmup=args.warmup//2, num_samples=args.samples//2, num_chains=args.chains)
        print("running mcmc: sampling prior...")
        mcmc.run(PRIOR_RNG, mass, spin, z, pedict, injdict, float(total_inj), nObs, obs_time, sample_prior=True) 
        prior = mcmc.get_samples()
        dd.io.save(f'{label}_prior_samples.h5', prior)
        
        kernel = NUTS(model)
        mcmc = MCMC(kernel, thinning=args.thinning, num_warmup=args.warmup, num_samples=args.samples, num_chains=args.chains)
        print("running mcmc: sampling posterior...")
        mcmc.run(MCMC_RNG, mass, spin, z, pedict, injdict, float(total_inj), nObs, obs_time, sample_prior=False) 
        mcmc.print_summary()
        posterior = mcmc.get_samples()
        dd.io.save(f'{label}_posterior_samples.h5', posterior)
        plot_params = [
            "detection_efficency",
            "lamb",
            "log_nEff_inj",
            "log_nEffs",
            "logBFs",
            "log_l",
            "mag_cs",
            "mag_tau",
            "mass_cs",
            "mass_tau",
            "q_cs", 
            "q_tau",
            "rate",
            "surveyed_hypervolume",
            "tilt_cs",
            "tilt_tau",
            "z_cs", 
            "z_tau",
        ]
        fig = az.plot_trace(az.from_numpyro(mcmc), var_names=plot_params);
        plt.savefig(f"{label}_trace_plot.png")
        del fig, mcmc, pedict, injdict, total_inj, obs_time
    else:
        print(f"loading prior and posterior samples from run with label: {label}...")
        prior = dd.io.load(f'{label}_prior_samples.h5')
        posterior = dd.io.load(f'{label}_posterior_samples.h5')
    
    print("plotting m1/z PPC...")
    fig = plot_m1_vs_z_ppc(posterior, nObs, min_m1, args.mmax, z.zmax)
    plt.savefig(f'{label}_m1_vs_z_ppc.png')
    del fig
    
    print("plotting m1 brontasaurus PPC...")
    fig = plot_ppc_brontosaurus(posterior, nObs, min_m1, args.mmax, z.zmax)
    plt.savefig(f'{label}_m1_ppc_brontosaurus.png')
    del fig
    
    print("calculating mass prior ppds...")
    prior_pm1s, prior_pqs, ms, qs = calculate_m1q_ppds(prior['mass_cs'], prior['q_cs'], BSplinePrimaryBSplineRatio, args.mass_knots, args.q_knots, mmin=args.mmin, m1mmin=min_m1, mmax=args.mmax)
    print("calculating mass posterior ppds...")
    pm1s, pqs, ms, qs = calculate_m1q_ppds(posterior['mass_cs'], posterior['q_cs'], BSplinePrimaryBSplineRatio, args.mass_knots, args.q_knots, mmin=args.mmin, m1mmin=min_m1, mmax=args.mmax)

    print("plotting mass distribution...")
    fig = plot_mass_dist(pm1s, pqs, ms, qs, mmin=min_m1, mmax=args.mmax, priors={'m1':prior_pm1s, 'q': prior_pqs});
    plt.savefig(f'{label}_mass_distribution.png')
    del fig

    print("calculating mag prior ppds...")
    prior_pmags, mags = calculate_iid_spin_ppds(prior['mag_cs'], BSplineIIDSpinMagnitudes, args.mag_knots, xmin=0,xmax=1)     
    print("calculating mag posterior ppds...")
    pmags, mags = calculate_iid_spin_ppds(posterior['mag_cs'], BSplineIIDSpinMagnitudes, args.mag_knots, xmin=0,xmax=1)
    
    print("calculating tilt prior ppds...")
    prior_ptilts, tilts = calculate_iid_spin_ppds(prior['tilt_cs'], BSplineIIDSpinTilts, args.tilt_knots, xmin=-1,xmax=1)
    print("calculating tilt posterior ppds...")
    ptilts, tilts = calculate_iid_spin_ppds(posterior['tilt_cs'], BSplineIIDSpinTilts, args.tilt_knots, xmin=-1,xmax=1)
        
    print("plotting spin distributions...")
    fig = plot_iid_spin_dist(pmags, ptilts, mags, tilts, priors={'mags':prior_pmags, 'tilts': prior_ptilts});
    plt.savefig(f'{label}_iid_component_spin_distribution.png')
    del fig
    
    print("calculating rate prior ppds...")
    prior_Rofz, zs = calculate_rate_of_z_ppds(prior['lamb'], prior['z_cs'], jnp.ones_like(prior['lamb']), z)
    print("calculating rate posterior ppds...")
    Rofz, zs = calculate_rate_of_z_ppds(posterior['lamb'], posterior['z_cs'], posterior['rate'], z)
    
    print("plotting R(z)...")
    fig = plot_rofz(Rofz, zs, prior=prior_Rofz)
    plt.savefig(f'{label}_rate_vs_z.png')
    del fig
    fig = plot_rofz(Rofz, zs, logx=True, prior=prior_Rofz)
    plt.savefig(f'{label}_rate_vs_z_logscale.png')
    del fig
    
    ppd_dict = {'dRdm1': pm1s, 'dRdq': pqs, 'm1s': ms, 'qs': qs, 'dRda': pmags, 'mags': mags, 'dRdct': ptilts, 'tilts': tilts, 'Rofz': Rofz, 'zs': zs}
    dd.io.save(f'{label}_ppds.h5', ppd_dict)
    prior_ppd_dict = {'pm1': prior_pm1s, 'pq': prior_pqs, 'pa': prior_pmags, 'pct': prior_ptilts, 'm1s': ms, 'qs': qs, 'mags': mags, 'tilts': tilts, 'Rofz': prior_Rofz, 'zs': zs}
    dd.io.save(f'{label}_prior_ppds.h5', prior_ppd_dict)


if __name__ == '__main__':
    main()