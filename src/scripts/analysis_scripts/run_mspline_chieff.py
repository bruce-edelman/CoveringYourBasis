#!/usr/bin/env python

import jax.numpy as jnp
import numpy as np
import numpyro
import deepdish as dd
from jax import random, device_put, devices
from numpyro import distributions as dist
from numpyro.infer import MCMC, NUTS
from gwheiro.models.msplines.seperable import MSplinePrimaryPowerlawRatio
from gwheiro.data_collection import load_injections, load_posterior_samples, convert_component_spin_injections_to_chieff, convert_component_spin_posteriors_to_chieff
from gwheiro.analysis_utils import heirarchical_likelihood
from gwheiro.models.msplines.single import MSplineChiEffective
import matplotlib.pyplot as plt
import arviz as az
from common import *
az.style.use("arviz-darkgrid")

#HARDCODED_KNOTS = np.array([-1.4, -1.2, -1.0, -0.8, -0.6, -0.5, -0.4, -0.32, -0.25, -0.19, -0.14, -0.1, -0.07, -0.05, -0.025, 
#                   0.0, 0.025, 0.05, 0.07, 0.1, 0.14, 0.19, 0.25, 0.32, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.4])

def setup_parser():
    parser = load_base_parser()
    return parser.parse_args()
    
    
def setup_effspin_MSpline_model(injdata, pedata, pmap, nknots, k=4):
    xmin, xmax = -1,1
    interior_knots = np.linspace(xmin, xmax, nknots-k+2)
    dx = interior_knots[1] - interior_knots[0]
    knots = np.concatenate([xmin-dx*np.arange(1,k)[::-1], interior_knots, xmax+dx*np.arange(1,k)])
    model = MSplineChiEffective(nknots, pedata[pmap['chi_eff']], injdata[pmap['chi_eff']], knots=knots)#HARDCODED_KNOTS)
    return {'model': model}


def setup(args):
    injections = load_injections(args.inj_file, spin=True)
    pe_samples, names = load_posterior_samples(args.data_dir, spin=True)
    param_names = ['mass_1', 'mass_ratio', 'redshift', 'a_1', 'a_2', 'cos_tilt_1', 'cos_tilt_2', 'prior']
    old_param_map = {p: i for i, p in enumerate(param_names)}
    injdata = np.array([injections[k] for k in param_names])
    pedata = np.array([[pe_samples[e][p] for e in names] for p in param_names])
    pedata, param_map = convert_component_spin_posteriors_to_chieff(pedata, old_param_map)
    injdata, param_map = convert_component_spin_injections_to_chieff(injdata, old_param_map)
    pedata = device_put(jnp.array(pedata), devices()[args.jax_gpu_device])
    injdata = device_put(jnp.array(injdata), devices()[args.jax_gpu_device])
    nObs = pedata.shape[1]
    total_inj = injections["total_generated"]
    obs_time = injections["analysis_time"]
    
    mass_dict, min_m1_interp = setup_mass_MSpline_model(injdata, pedata, param_map, args.mass_knots, mmax=args.mmax)
    z_dict = setup_redshift_model(injdata, pedata, param_map)
    
    prior_dict = {'pe': pedata[param_map['prior']], 'inj': injdata[param_map['prior']]}
    spin_dict = setup_effspin_MSpline_model(injdata, pedata, param_map, args.chieff_knots)    
    print(f"{len(z_dict['inj'])} found injections out of {total_inj} total")
    print(f"Observed {nObs} events, each with {z_dict['pe'].shape[1]} samples, over an observing time of {obs_time} yrs")    
    return mass_dict, spin_dict, z_dict, prior_dict, total_inj, nObs, obs_time, min_m1_interp


def model(mass_dict, spin_dict, z_dict, prior_dict, total_inj, Nobs, Tobs, args):
    z_model = z_dict['model']
    m1_pen_degree = 2
    mass_model = mass_dict['model']
    mass_knots = mass_model.primary_model.nknots
    spin_model = spin_dict['model']
    chi_eff_knots = spin_model.nknots
    
    mass_cs = numpyro.sample('mass_cs', dist.Exponential(mass_knots), sample_shape=(mass_knots,))
    #mass_cs = numpyro.sample('mass_cs', dist.Uniform(0, 10*mass_knots), sample_shape=(mass_knots,))
    mass_tau = numpyro.deterministic("mass_tau",  mixture_smoothing_parameter("mass", n_mix=24, log10bmin=-6, log10bmax=-2))
    #mass_Lambda = get_adaptive_Lambda("mass", mass_knots, degree=m1_pen_degree, omega=0.1)
    numpyro.factor("mass_log_smoothing_penalty", calculate_penalty(mass_cs, mass_tau, degree=m1_pen_degree))
    
    if args.dirichlet:
        chi_eff_cs = numpyro.sample('chi_eff_cs', dist.Dirichlet(jnp.ones(chi_eff_knots)))
    else:
        chi_eff_cs = numpyro.sample('chi_eff_cs', dist.Exponential(chi_eff_knots), sample_shape=(chi_eff_knots,))
        chi_eff_tau = numpyro.deterministic("chi_eff_tau",  mixture_smoothing_parameter("chi_eff", n_mix=5, log10bmin=-4, log10bmax=-1))
        #chi_eff_tau = numpyro.sample("chi_eff_tau", dist.Gamma(1.0, 1e-3)) 
        numpyro.factor("chi_eff_log_smoothing_penalty", calculate_penalty(chi_eff_cs, chi_eff_tau, degree=m1_pen_degree))
    beta = numpyro.sample("beta", dist.Normal(0,2))
    lamb = numpyro.sample("lamb", dist.Normal(0,2))
    mmin = args.mmin
    
    def get_weights(m1,q,z,prior):
        p_m1q = mass_model(m1, q, beta, mmin, mass_cs)
        p_chieff= spin_model(len(z.shape), chi_eff_cs)
        p_z = z_model(z, lamb)
        wts = p_m1q*p_chieff*p_z/prior
        return jnp.where(jnp.isnan(wts) | jnp.isinf(wts), 0, wts)

    peweights = get_weights(mass_dict['pe']['mass_1'], mass_dict['pe']['mass_ratio'], z_dict['pe'], prior_dict['pe'])
    injweights = get_weights(mass_dict['inj']['mass_1'], mass_dict['inj']['mass_ratio'], z_dict['inj'], prior_dict['inj'])

    heirarchical_likelihood(peweights, injweights, total_inj=total_inj, Nobs=Nobs, Tobs=Tobs, 
                            surv_hypervolume_fct=z_model.normalization, lamb=lamb, marginalize_selection=False)
    

def main():
    args = setup_parser()
    #args.chieff_knots = len(HARDCODED_KNOTS)-4
    if args.dirichlet:
        label = f'results/paper/mspline_{args.mass_knots}m1_{args.chieff_knots}chieff_dirchprior_powerlaw_q_z_fitlamb'
    else:
        label = f'results/paper/mspline_{args.mass_knots}m1_{args.chieff_knots}chieff_smoothprior_powerlaw_q_z_fitlamb'
    RNG = random.PRNGKey(0)
    MCMC_RNG, RNG = random.split(RNG)
    kernel = NUTS(model)
    mcmc = MCMC(kernel, num_warmup=args.warmup, num_samples=args.samples, num_chains=args.chains, chain_method='sequential', progress_bar=True)
    mass, spin, z, prior, total_inj, nObs, obs_time, m1min = setup(args)
    mcmc.run(MCMC_RNG, mass, spin, z, prior, float(total_inj), nObs, obs_time, args) 
    mcmc.print_summary()
    posterior = mcmc.get_samples()
    pm1s, pqs, ms, qs = calculate_m1q_ppds(posterior['mass_cs'], posterior['rate'], posterior['beta'], MSplinePrimaryPowerlawRatio, nknots=args.mass_knots, mmin=args.mmin, m1mmin=m1min, mmax=args.mmax)
    pchieffs, chieffs = calculate_chi_ppds(posterior['chi_eff_cs'], posterior['rate'], MSplineChiEffective, nknots=args.chieff_knots, xmin=-1,xmax=1)
    try:
        Rofz, zs = calculate_rate_of_z_ppds(posterior['lamb'], posterior['rate'], z['model'])
    except KeyError:
        Rofz, zs = calculate_rate_of_z_ppds(jnp.zeros_like(posterior['rate']), posterior['rate'], z['model'])
    ppd_dict = {'dRdm1': pm1s, 'dRdq': pqs, 'm1s': ms, 'qs': qs, 'dRdchieff': pchieffs, 'chieffs': chieffs, 'Rofz': Rofz, 'zs': zs}
    dd.io.save(f'{label}_posterior_samples.h5', posterior)
    dd.io.save(f'{label}_ppds.h5', ppd_dict)
    idata = az.from_numpyro(mcmc)
    az.plot_trace(idata)
    plt.savefig(f'{label}_trace_plot.png')
    fig = plot_mass_dist(pm1s, pqs, ms, qs, mmin=m1min, mmax=args.mmax);
    plt.savefig(f'{label}_mass_distribution.png')
    fig = plot_chieff_dist(pchieffs, chieffs);
    plt.savefig(f'{label}_effective_spin_distribution.png')


if __name__ == '__main__':
    main()