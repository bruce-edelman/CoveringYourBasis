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
from utils import LogXLogYBSpline, LogYBSpline, MSplineIndependentSpinTilts, MSplineIndependentSpinMagnitudes, MSplinePrimaryMSplineRatio, MSplineIIDSpinMagnitudes, MSplineIIDSpinTilts
import deepdish as dd
from tqdm import trange
import paths
import warnings
warnings.filterwarnings('ignore')


def draw_chieff_samples_gwpop(result, magmodel, tiltmodel, massmodel, ndraws=500, nsamples=2000):
    mags = np.linspace(0,1,1500)
    a1s, a2s = np.meshgrid(mags, mags)
    m1s = np.linspace(2,100,2000)
    qs = np.linspace(0.05, 1, 1000)
    mms, qqs = np.meshgrid(m1s, qs)
    ctilts = np.linspace(-1,1,1500)
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

def draw_chieff_samples(posterior, massmodel, magmodel, tiltmodel, massnknots, qnknots, magnknots, tiltnknots, ndraws=500, nsamples=2000, iid=False):
    mags = jnp.linspace(0,1,1500)
    a1s, a2s = jnp.meshgrid(mags, mags)
    m1s = jnp.linspace(3,100,2000)
    qs = jnp.linspace(3.0/100.0, 1, 1000)
    mms, qqs = jnp.meshgrid(m1s, qs)
    ctilts = jnp.linspace(-1,1,1500)
    ct1s, ct2s = jnp.meshgrid(ctilts, ctilts)
    mmin = 5.0
    mmax = 100.0
    
    mass_model = massmodel(massnknots, qnknots, mms, m1s, qqs, qs, m1min=mmin, m2min=5.0, mmax=mmax, basis_m=LogXLogYBSpline, basis_q=LogYBSpline, normalize=True)
    
    if iid:
        mag_model = magmodel(magnknots, a1s, a2s, mags, mags, basis=LogYBSpline, normalize=True)
        tilt_model = tiltmodel(tiltnknots, ct1s, ct2s, ctilts, ctilts, basis=LogYBSpline, normalize=True)
    else:
        mag_model = magmodel(magnknots, magnknots, a1s, a2s, mags, mags, basis=LogYBSpline, normalize=True)
        tilt_model = tiltmodel(tiltnknots, tiltnknots, ct1s, ct2s, ctilts, ctilts, basis=LogYBSpline, normalize=True)
    
    samples = {'a_1': np.empty((ndraws, nsamples)), 'a_2': np.empty((ndraws, nsamples)), 'cos_tilt_1': np.empty((ndraws, nsamples)),
               'cos_tilt_2': np.empty((ndraws, nsamples)), 'mass_1': np.empty((ndraws, nsamples)), 'mass_ratio': np.empty((ndraws, nsamples))}
    
    @jit
    def calc_pdfs(mcs, qcs, a1cs, a2cs, t1cs, t2cs):
        p_mq = mass_model(2, mcs, qcs)
        p_mq = jnp.where(jnp.isinf(p_mq) | jnp.isnan(p_mq), 0, p_mq)
        p_mq = jnp.where(jnp.greater(mms, mmax) | jnp.less(mms, mmin) | jnp.less(mms*qqs, mmin), 0, p_mq)
        p_m = jnp.trapz(p_mq, qs, axis=0)
        p_q = jnp.trapz(p_mq, m1s, axis=1)
        pa1 = mag_model.primary_model(1, a1cs)
        pa2 = mag_model.secondary_model(1, a2cs)
        pt1 = tilt_model.primary_model(1, t1cs)
        pt2 = tilt_model.secondary_model(1, t2cs)
        #pa1 = jnp.where(jnp.isinf(pa1) | jnp.isnan(pa1) | jnp.less(pa1,0), 0, pa1)
        #pa2 = jnp.where(jnp.isinf(pa2) | jnp.isnan(pa2) | jnp.less(pa2,0), 0, pa2)
        #pt1 = jnp.where(jnp.isinf(pt1) | jnp.isnan(pt1) | jnp.less(pt1,0), 0, pt1)
        #pt2 = jnp.where(jnp.isinf(pt2) | jnp.isnan(pt2) | jnp.less(pt2,0), 0, pt2)
        return p_m, p_q, pa1, pa2, pt1, pt2
    
    for i in trange(ndraws):
        idx = np.random.choice(len(posterior))
        if iid:
            pm1, pq, pa1, pa2, pc1, pc2 = calc_pdfs(posterior['mass_cs'][idx], posterior['q_cs'][idx], 
                                                    posterior['mag_cs'][idx], posterior['mag_cs'][idx], 
                                                    posterior['tilt_cs'][idx], posterior['tilt_cs'][idx])
        else:
            pm1, pq, pa1, pa2, pc1, pc2 = calc_pdfs(posterior['mass_cs'][idx], posterior['q_cs'][idx], 
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
    mspl_post = dd.io.load(paths.data / "bsplines_64m1_18q_ind18mag_ind18tilt_pl18z_posterior_samples.h5")
    chi_effs_mspl = draw_chieff_samples(mspl_post, MSplinePrimaryMSplineRatio, MSplineIndependentSpinMagnitudes, MSplineIndependentSpinTilts, 
                                        64, 18, 18, 18)
    pchieffs, chieffs = chi_eff_kde_ppd(chi_effs_mspl)
    mspl_post_iid = dd.io.load(paths.data / "bsplines_64m1_18q_iid18mag_iid18tilt_pl18z_posterior_samples.h5")
    chi_effs_mspl_iid = draw_chieff_samples(mspl_post_iid, MSplinePrimaryMSplineRatio, MSplineIIDSpinMagnitudes, MSplineIIDSpinTilts, 
                                            64, 18, 18, 18, iid=True)
    pchieffs_iid, chieffs_iid = chi_eff_kde_ppd(chi_effs_mspl_iid)
    o3b_default_spin_result = read_in_result(paths.data / "o1o2o3_mass_c_iid_mag_iid_tilt_powerlaw_redshift_result.json")
    chi_effs_o3b_default = draw_chieff_samples_gwpop(o3b_default_spin_result, DefaultSpinMagModel, DefaultSpinTiltModel, MassModel)
    pchieffs_def, chieffs_def = chi_eff_kde_ppd(chi_effs_o3b_default)
    datadict = {'BSplineInd': {'pchieff': pchieffs, 'chieffs': chieffs}, 
                'BSplineIID': {'pchieff': pchieffs_iid, 'chieffs': chieffs_iid}, 
                'Default': {'pchieff':pchieffs_def, 'chieffs':chieffs_def}}
    dd.io.save(paths.data / "chi_eff_ppds.h5", datadict)

if __name__ == "__main__":
    main()