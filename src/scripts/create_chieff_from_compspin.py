import numpy as np
import jax.numpy as jnp
from scipy.stats import gaussian_kde
from gwpopulation.cupy_utils import trapz
from gwpopulation.conversions import convert_to_beta_parameters
from gwpopulation.models.mass import SinglePeakSmoothedMassDistribution
from gwpopulation.models.spin import iid_spin_orientation_gaussian_isotropic, iid_spin_magnitude_beta
from bilby.core.result import read_in_result
from bilby.hyper.model import Model
from jax import jit
from utils import MSplineIndependentSpinTilts, MSplineIndependentSpinMagnitudes, MSplinePrimaryPowerlawRatio, MSplineIIDSpinMagnitudes, MSplineIIDSpinTilts
import deepdish as dd
from tqdm import trange
import paths
import warnings
warnings.filterwarnings('ignore')


def draw_chieff_samples_gwpop(result, magmodel, tiltmodel, massmodel, ndraws=500, nsamples=2000):
    mags = np.linspace(0,1,2000)
    a1s, a2s = np.meshgrid(mags, mags)
    m1s = np.linspace(2,100,200)
    qs = np.linspace(0.05, 1, 2000)
    mms, qqs = np.meshgrid(m1s, qs)
    ctilts = np.linspace(-1,1,2000)
    ct1s, ct2s = np.meshgrid(ctilts, ctilts)
    
    magdata = {'a_1': a1s, 'a_2': a2s}
    tiltdata = {'cos_tilt_1': ct1s, 'cos_tilt_2': ct2s}
    massdata = {'mass_1': mms, 'mass_ratio': qqs}
    
    samples = {'a_1': np.empty((ndraws, nsamples)), 'a_2': np.empty((ndraws, nsamples)), 'cos_tilt_1': np.empty((ndraws, nsamples)),
               'cos_tilt_2': np.empty((ndraws, nsamples)), 'mass_1': np.empty((ndraws, nsamples)), 'mass_ratio': np.empty((ndraws, nsamples))}
    
    idxs = np.random.choice(len(result.posterior), size=ndraws, replace=False)

    for i in trange(ndraws):
        samp  = result.posterior.iloc[idxs[i]]
        if 'mu_chi' in samp:
            samp, _ = convert_to_beta_parameters(samp)
        magmodel.parameters.update(samp)
        tiltmodel.parameters.update(samp)
        massmodel.parameters.update(samp)
        pa1pa2 = magmodel.prob(magdata)
        pa1 = trapz(pa1pa2, mags, axis=0)
        pa2 = trapz(pa1pa2, mags, axis=1)
        pct1ct2 = tiltmodel.prob(tiltdata)
        pc1 = trapz(pct1ct2, ctilts, axis=0)
        pc2 = trapz(pct1ct2, ctilts, axis=1)
        pm1pq = massmodel.prob(massdata)
        pm1 = trapz(pm1pq, qs, axis=0)
        pq = trapz(pm1pq, m1s, axis=1)
        samples['a_1'][i,:] = np.random.choice(mags, p=pa1/np.sum(pa1), size=nsamples)
        samples['a_2'][i,:] = np.random.choice(mags, p=pa2/np.sum(pa2), size=nsamples)
        samples['cos_tilt_1'][i,:] = np.random.choice(ctilts, p=pc1/np.sum(pc1), size=nsamples)
        samples['cos_tilt_2'][i,:] = np.random.choice(ctilts, p=pc2/np.sum(pc2), size=nsamples)
        samples['mass_1'][i,:] = np.random.choice(m1s, p=pm1/np.sum(pm1), size=nsamples)
        samples['mass_ratio'][i,:] = np.random.choice(qs, p=pq/np.sum(pq), size=nsamples)
    return chi_eff(samples)

def chi_eff(samples):
    return (samples['a_1']*samples['cos_tilt_1'] + samples['mass_ratio']*samples['a_2']*samples['cos_tilt_2']) / (1.0 + samples['mass_ratio'])

def draw_chieff_samples(posterior, massmodel, magmodel, tiltmodel, massnknots, magnknots, tiltnknots, k=4, ndraws=500, nsamples=2000, iid=False):
    mags = jnp.linspace(0,1,2000)
    a1s, a2s = jnp.meshgrid(mags, mags)
    m1s = jnp.linspace(2,100,200)
    qs = jnp.linspace(0.05, 1, 2000)
    mms, qqs = jnp.meshgrid(m1s, qs)
    ctilts = jnp.linspace(-1,1,2000)
    ct1s, ct2s = jnp.meshgrid(ctilts, ctilts)
    mmin = 6.5
    mmax = 100.0
    
    interior_logknots = np.linspace(np.log10(mmin), np.log10(mmax), massnknots-k+2)
    dx = interior_logknots[1] - interior_logknots[0]
    logknots = np.concatenate([np.log10(mmin)-dx*np.arange(1,k)[::-1], interior_logknots, np.log10(mmax)+dx*np.arange(1,k)])
    knots = 10**(logknots)
    mass_model = massmodel(massnknots, mms, m1s, knots=knots)
    
    xmin, xmax = 0,1
    interior_knots = np.linspace(xmin, xmax, magnknots-k+2)
    dx = interior_knots[1] - interior_knots[0]
    magknots = np.concatenate([xmin-dx*np.arange(1,k)[::-1], interior_knots, xmax+dx*np.arange(1,k)])
    xmin, xmax = -1,1
    interior_knots = np.linspace(xmin, xmax, tiltnknots-k+2)
    dx = interior_knots[1] - interior_knots[0]
    tiltknots = np.concatenate([xmin-dx*np.arange(1,k)[::-1], interior_knots, xmax+dx*np.arange(1,k)])
    if iid:
        mag_model = magmodel(magnknots, a1s, a2s, mags, mags, knots=magknots)
        tilt_model = tiltmodel(tiltnknots, ct1s, ct2s, ctilts, ctilts, knots=tiltknots)
    else:
        mag_model = magmodel(magnknots, magnknots, a1s, a2s, mags, mags, knots1=magknots, knots2=magknots)
        tilt_model = tiltmodel(tiltnknots, tiltnknots, ct1s, ct2s, ctilts, ctilts, knots1=tiltknots, knots2=tiltknots)
    
    samples = {'a_1': np.empty((ndraws, nsamples)), 'a_2': np.empty((ndraws, nsamples)), 'cos_tilt_1': np.empty((ndraws, nsamples)),
               'cos_tilt_2': np.empty((ndraws, nsamples)), 'mass_1': np.empty((ndraws, nsamples)), 'mass_ratio': np.empty((ndraws, nsamples))}
    
    @jit
    def calc_pdfs(mcs, bet, a1cs, a2cs, t1cs, t2cs):
        p_mq = mass_model(mms, qqs, bet, mmin, mcs)
        p_mq = jnp.where(jnp.isinf(p_mq) | jnp.isnan(p_mq), 0, p_mq)
        p_mq = jnp.where(jnp.greater(mms, mmax) | jnp.less(mms, mmin) | jnp.less(mms*qqs, mmin), 0, p_mq)
        p_m = jnp.trapz(p_mq, qs, axis=0)
        p_q = jnp.trapz(p_mq, m1s, axis=1)
        pa1 = mag_model.primary_model(1, a1cs)
        pa2 = mag_model.secondary_model(1, a2cs)
        pt1 = tilt_model.primary_model(1, t1cs)
        pt2 = tilt_model.secondary_model(1, t2cs)
        return p_m, p_q, pa1, pa2, pt1, pt2
    
    for i in trange(ndraws):
        idx = np.random.choice(len(posterior))
        if iid:
            pm1, pq, pa1, pa2, pc1, pc2 = calc_pdfs(posterior['mass_cs'][idx], posterior['beta'][idx], 
                                                    posterior['mag_cs'][idx], posterior['mag_cs'][idx], 
                                                    posterior['tilt_cs'][idx], posterior['tilt_cs'][idx])
        else:
            pm1, pq, pa1, pa2, pc1, pc2 = calc_pdfs(posterior['mass_cs'][idx], posterior['beta'][idx], 
                                                    posterior['mag_cs'][idx,:,0], posterior['mag_cs'][idx,:,1], 
                                                    posterior['tilt_cs'][idx,:,0], posterior['tilt_cs'][idx,:,1])
        pm1 = np.array(pm1)
        pq = np.array(pq)
        pa1 = np.array(pa1)
        pa2 = np.array(pa2)
        pc1 = np.array(pc1)
        pc2 = np.array(pc2)
        
        samples['a_1'][i,:] = np.random.choice(mags, p=pa1/np.sum(pa1), size=nsamples)
        samples['a_2'][i,:] = np.random.choice(mags, p=pa2/np.sum(pa2), size=nsamples)
        samples['cos_tilt_1'][i,:] = np.random.choice(ctilts, p=pc1/np.sum(pc1), size=nsamples)
        samples['cos_tilt_2'][i,:] = np.random.choice(ctilts, p=pc2/np.sum(pc2), size=nsamples)
        samples['mass_1'][i,:] = np.random.choice(m1s, p=pm1/np.sum(pm1), size=nsamples)
        samples['mass_ratio'][i,:] = np.random.choice(qs, p=pq/np.sum(pq), size=nsamples)
    
    return chi_eff(samples)   
    
    

def chi_eff_kde_ppd(samples):
    x = np.linspace(-1,1,1000)
    pxs = []
    for ii in range(samples.shape[0]):
        k = gaussian_kde(samples[ii])
        px = k(x)
        pxs.append(px)
    return np.array(pxs), x
        

def main():
    DefaultSpinTiltModel = Model([iid_spin_orientation_gaussian_isotropic])
    DefaultSpinMagModel = Model([iid_spin_magnitude_beta])
    MassModel = Model([SinglePeakSmoothedMassDistribution(mmin=2.0, mmax=100.0)])
    mspl_post = dd.io.load(paths.data / "mspline_50m1_16ind_compspins_smoothprior_powerlaw_q_z_posterior_samples.h5")
    chi_effs_mspl = draw_chieff_samples(mspl_post, MSplinePrimaryPowerlawRatio, MSplineIndependentSpinMagnitudes, MSplineIndependentSpinTilts, 
                                        50, 16, 16)
    pchieffs, chieffs = chi_eff_kde_ppd(chi_effs_mspl)
    mspl_post_iid = dd.io.load(paths.data / "mspline_50m1_16iid_compspins_smoothprior_powerlaw_q_z_posterior_samples.h5")
    chi_effs_mspl_iid = draw_chieff_samples(mspl_post_iid, MSplinePrimaryPowerlawRatio, MSplineIIDSpinMagnitudes, MSplineIIDSpinTilts, 
                                            50, 16, 16, iid=True)
    pchieffs_iid, chieffs_iid = chi_eff_kde_ppd(chi_effs_mspl_iid)
    o3b_default_spin_result = read_in_result(paths.data / "o1o2o3_mass_c_iid_mag_iid_tilt_powerlaw_redshift_result.json")
    chi_effs_o3b_default = draw_chieff_samples_gwpop(o3b_default_spin_result, DefaultSpinMagModel, DefaultSpinTiltModel, MassModel)
    pchieffs_def, chieffs_def = chi_eff_kde_ppd(chi_effs_o3b_default)
    datadict = {'MSplineInd': {'pchieff': pchieffs, 'chieffs': chieffs}, 
                'MSplineIID': {'pchieff': pchieffs_iid, 'chieffs': chieffs_iid}, 
                'Default': {'pchieff':pchieffs_def, 'chieffs':chieffs_def}}
    dd.io.save(paths.data / "chi_eff_ppds.h5", datadict)

if __name__ == "__main__":
    main()