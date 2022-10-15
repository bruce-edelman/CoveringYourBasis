#!/usr/bin/env python

import json2latex
import json
import paths
import numpy as np
import deepdish as dd
from scipy.integrate import cumtrapz
from utils import load_iid_mag_ppd, load_iid_tilt_ppd, load_ind_mag_ppd, load_ind_tilt_ppd, load_o3b_paper_run_masspdf, load_mass_ppd, load_iid_posterior, load_ind_posterior, load_o3b_posterior

def round_sig(f, sig=2):
    max10exp = np.floor(np.log10(abs(f))) + 1
    if max10exp < float(sig):
        return float(('%.' + str(sig) + 'g') % f)
    else:
        return int(float(('%.' + str(sig) + 'g') % f))

def save_param_cred_intervals(param_data):
    return  {'median': round_sig(np.median(param_data)), 
             'error plus':round_sig(np.percentile(param_data, 95)-np.mean(param_data)), 
             'error minus': round_sig(np.median(param_data)-np.percentile(param_data, 5)),
             '5th percentile': round_sig(np.percentile(param_data, 5)), 
             '95th percentile': round_sig(np.percentile(param_data, 95)), 
             '10th percentile': round_sig(np.percentile(param_data, 10)),
             '90th percentile': round_sig(np.percentile(param_data, 90))}

def get_percentile(pdfs, xs, perc):
    x = []
    for m in pdfs:                                                                                                                                                                        
        i = len(m)
        cumulative_prob = cumtrapz(m, xs, initial = 0)
        init_prob = cumulative_prob[-1]
        prob = init_prob
        final_prob = init_prob * perc / 100.0                                                                                                                                              
        while prob > (final_prob):
            i -= 1
            prob = cumulative_prob[i]                                                                                                                                                         
        x.append(xs[i])
    return np.array(x)

def BSplineMassMacros():
    print("Saving Mass Distribution Macros...")
    ms, m_pdfs, _, _ = load_mass_ppd()
    m1s = get_percentile(m_pdfs, ms, 1)
    m99s = get_percentile(m_pdfs, ms, 99)
    m75s = get_percentile(m_pdfs, ms, 75)
    plpeak_mpdfs, _, plpeak_ms, _ = load_o3b_paper_run_masspdf(paths.data / 'o1o2o3_mass_c_iid_mag_iid_tilt_powerlaw_redshift_mass_data.h5')
    plpeak_m1s = get_percentile(plpeak_mpdfs, plpeak_ms, 1) 
    plpeak_m99s = get_percentile(plpeak_mpdfs, plpeak_ms, 99)
    plpeak_m75s = get_percentile(plpeak_mpdfs, plpeak_ms, 75)  
    ps_mpdfs, _, ps_ms, _ = load_o3b_paper_run_masspdf(paths.data / 'spline_20n_mass_m_iid_mag_iid_tilt_powerlaw_redshift_mass_data.h5')
    ps_m1s = get_percentile(plpeak_mpdfs, plpeak_ms, 1) 
    ps_m99s = get_percentile(ps_mpdfs, ps_ms, 99)
    ps_m75s = get_percentile(ps_mpdfs, ps_ms, 75)  
    return {'PLPeak': {'m_1percentile': save_param_cred_intervals(plpeak_m1s), 'm_75percentile': save_param_cred_intervals(plpeak_m75s), 'm_99percentile': save_param_cred_intervals(plpeak_m99s)}, 
            'BSpline': {'m_1percentile': save_param_cred_intervals(m1s), 'm_75percentile': save_param_cred_intervals(m75s), 'm_99percentile': save_param_cred_intervals(m99s)}, 
            'PLSpline': {'m_1percentile': save_param_cred_intervals(ps_m1s), 'm_75percentile': save_param_cred_intervals(ps_m75s), 'm_99percentile': save_param_cred_intervals(ps_m99s)}}

def BSplineIIDSpinMacros():
    print("Saving BSpline IID Component Spin macros...")
    xs, ct_pdfs = load_iid_tilt_ppd()
    peak_tilts = []
    gamma_fracs = []
    frac_neg_cts = []
    for jj in range(len(ct_pdfs)):
        ct_pdfs[jj,:] /= np.trapz(ct_pdfs[jj,:], xs)
        peak_tilts.append(xs[np.argmax(ct_pdfs[jj,:])])
        gam = ct_pdfs[jj,xs>=0.9] / ct_pdfs[jj, xs<=-0.9]
        gamma_fracs.append(gam)
        neg = xs <= 0
        frac_neg_cts.append(np.trapz(ct_pdfs[jj,neg], x=xs[neg]))
    peak_tilts = np.array(peak_tilts)
    gamma_fracs = np.array(gamma_fracs)
    frac_neg_cts = np.array(frac_neg_cts)
    posts = load_iid_posterior()
    mags, a_pdfs = load_iid_mag_ppd()
    peak_mags = []
    for jj in range(len(a_pdfs)):
        peak_mags.append(mags[np.argmax(a_pdfs[jj,:])])
    peak_mags = np.array(peak_mags)
    return {'a_90percentile': save_param_cred_intervals(get_percentile(a_pdfs, mags, 90)),
            'peak_a': save_param_cred_intervals(peak_mags),
            'lamb': save_param_cred_intervals(posts['lamb']),
            'local_rate': save_param_cred_intervals(posts['rate']),
            'peakCosTilt': save_param_cred_intervals(peak_tilts), 
            'log10gammaFrac': save_param_cred_intervals(np.log10(gamma_fracs)), 
            'negFrac': save_param_cred_intervals(frac_neg_cts)}
    
    
def DefaultSpinMacros():
    print("Saving Default Component Spin macros...")
    xs = np.linspace(-1, 1, 1000)
    tilt_data = dd.io.load(paths.data / 'o1o2o3_mass_c_iid_mag_iid_tilt_powerlaw_redshift_orientation_data.h5')
    lines = dict()
    for key in tilt_data["lines"].keys():
        lines[key] = tilt_data["lines"][key][()]
        for ii in range(len(lines[key])):
            lines[key][ii] /= np.trapz(lines[key][ii], xs)
    ct_pdfs = lines['cos_tilt_1']
    peak_tilts = []
    gamma_fracs = []
    frac_neg_cts = []
    for jj in range(len(ct_pdfs)):
        ct_pdfs[jj,:] /= np.trapz(ct_pdfs[jj,:], xs)
        peak_tilts.append(xs[np.argmax(ct_pdfs[jj,:])])
        gam = ct_pdfs[jj,xs>=0.9] / ct_pdfs[jj, xs<=-0.9]
        gamma_fracs.append(gam)
        neg = xs <= 0
        frac_neg_cts.append(np.trapz(ct_pdfs[jj,neg], x=xs[neg]))
    peak_tilts = np.array(peak_tilts)
    gamma_fracs = np.array(gamma_fracs)
    frac_neg_cts = np.array(frac_neg_cts)
    mags = np.linspace(0, 1, 1000)
    mag_data = dd.io.load(paths.data / 'o1o2o3_mass_c_iid_mag_iid_tilt_powerlaw_redshift_magnitude_data.h5')
    lines = dict()
    peak_mags = []
    for key in mag_data["lines"].keys():
        lines[key] = mag_data["lines"][key][()]
        for ii in range(len(lines[key])):
            lines[key][ii] /= np.trapz(lines[key][ii], xs)    
    a_pdfs = lines['a_1']
    for jj in range(len(a_pdfs)):
        peak_mags.append(mags[np.argmax(a_pdfs[jj,:])])
    peak_mags = np.array(peak_mags)
    posts = load_o3b_posterior('o1o2o3_mass_c_iid_mag_iid_tilt_powerlaw_redshift_result.json')
            
    return {'a_90percentile': save_param_cred_intervals(get_percentile(a_pdfs, mags, 90)),
            'peak_a': save_param_cred_intervals(peak_mags),
            'lamb': save_param_cred_intervals(posts['lamb']),
            'local_rate': save_param_cred_intervals(posts['rate']),
            'peakCosTilt': save_param_cred_intervals(peak_tilts), 
            'log10gammaFrac': save_param_cred_intervals(np.log10(gamma_fracs)), 
            'negFrac': save_param_cred_intervals(frac_neg_cts)}
    

def BSplineIndSpinMacros():
    print("Saving BSpline Independent Component Spin macros...")
    xs, ct1_pdfs, ct2_pdfs = load_ind_tilt_ppd()
    peak_tilts1 = []
    peak_tilts2 = []
    gamma_fracs1 = []
    gamma_fracs2 = []
    frac_neg_cts1 = []
    frac_neg_cts2 = []
    for jj in range(len(ct1_pdfs)):
        ct1_pdfs[jj,:] /= np.trapz(ct1_pdfs[jj,:], xs)
        ct2_pdfs[jj,:] /= np.trapz(ct2_pdfs[jj,:], xs)
        peak_tilts1.append(xs[np.argmax(ct1_pdfs[jj,:])])
        peak_tilts2.append(xs[np.argmax(ct2_pdfs[jj,:])])
        gam1 = ct1_pdfs[jj,xs>=0.9] / ct1_pdfs[jj, xs<=-0.9]
        gamma_fracs1.append(gam1)
        gam2 = ct2_pdfs[jj,xs>=0.9] / ct2_pdfs[jj, xs<=-0.9]
        gamma_fracs2.append(gam2)
        neg = xs <= 0
        frac_neg_cts1.append(np.trapz(ct1_pdfs[jj,neg], x=xs[neg]))
        frac_neg_cts2.append(np.trapz(ct2_pdfs[jj,neg], x=xs[neg]))

    peak_tilts1 = np.array(peak_tilts1)
    peak_tilts2 = np.array(peak_tilts2)
    gamma_fracs1 = np.array(gamma_fracs1)
    gamma_fracs2 = np.array(gamma_fracs2)
    frac_neg_cts1 = np.array(frac_neg_cts1)
    frac_neg_cts2 = np.array(frac_neg_cts2)
    posts = load_ind_posterior()
    a1_pdfs, a2_pdfs, mags, mags = load_ind_mag_ppd()
    peak_mag1s = []
    peak_mag2s = []
    for jj in range(len(a1_pdfs)):
        peak_mag1s.append(mags[np.argmax(a1_pdfs[jj,:])])
        peak_mag2s.append(mags[np.argmax(a2_pdfs[jj,:])])
    peak_mag2s = np.array(peak_mag2s)
    peak_mag1s = np.array(peak_mag1s)
    return {'lamb': save_param_cred_intervals(posts['lamb']),
            'local_rate': save_param_cred_intervals(posts['rate']),
            'a1_90percentile': save_param_cred_intervals(get_percentile(a1_pdfs, mags, 90)),
            'a2_90percentile': save_param_cred_intervals(get_percentile(a2_pdfs, mags, 90)),
            'peak_a1': save_param_cred_intervals(peak_mag1s),
            'peak_a2': save_param_cred_intervals(peak_mag2s),
            'peakCosTilt1': save_param_cred_intervals(peak_tilts1), 'peakCosTilt2': save_param_cred_intervals(peak_tilts2), 
            'log10gammaFrac1': save_param_cred_intervals(np.log10(gamma_fracs1)), 'log10gammaFrac2': save_param_cred_intervals(np.log10(gamma_fracs2)), 
            'negFrac1': save_param_cred_intervals(frac_neg_cts1), 'negFrac2': save_param_cred_intervals(frac_neg_cts2)}
    
def chi_eff():
    print("Saving ChiEffective macros...")
    ppds = dd.io.load(paths.data / "chi_eff_ppds.h5")
    default = ppds["Default"]
    BSplined = ppds["BSplineInd"]
    mspliid = ppds['BSplineIID']
    #msplchieff = dd.io.load(paths.data / "BSpline_50m1_24chieff_smoothprior_powerlaw_q_z_fitlamb_ppds.h5")
    below_0 = {'default': [], 'iid': [], 'ind': [], 'gaussian': []}#'chieff': [], 
    below_m0p3 = {'default': [], 'iid': [], 'ind': [], 'gaussian': []}#'chieff': [], 
    maxchis = {'default': [], 'iid': [], 'ind': [], 'gaussian': []}#'chieff': [], 
    with open(paths.data / "gaussian-spin-xeff-xp-ppd-data.json",'r') as jf:
        gaussian_data = json.load(jf)
        gauschieff = {'pchieff': np.array(gaussian_data['chi_eff_pdfs']), 'chieffs': np.array(gaussian_data['chi_eff_grid'])}
    #for i in range(1500):
    #    norm = np.trapz(msplchieff['dRdchieff'][i, :], x=msplchieff['chieffs'])
    #    below_m0p3['chieff'].append(np.trapz(1./norm*msplchieff['dRdchieff'][i, msplchieff['chieffs']<-0.3], x=msplchieff['chieffs'][msplchieff['chieffs']<-0.3]))
    #    below_0['chieff'].append(np.trapz(1./norm*msplchieff['dRdchieff'][i, msplchieff['chieffs']<0.0], x=msplchieff['chieffs'][msplchieff['chieffs']<0.0]))
    #    maxchis['chieff'].append(msplchieff['chieffs'][np.argmax(msplchieff['dRdchieff'][i])])
    for j in range(500):
        for k,v in zip(['default', 'ind', 'iid', 'gaussian'], [default, BSplined, mspliid, gauschieff]):
            below_m0p3[k].append(np.trapz(v['pchieff'][j,v['chieffs']<-0.3], x=v['chieffs'][v['chieffs']<-0.3]))
            below_0[k].append(np.trapz(v['pchieff'][j,v['chieffs']<0.0], x=v['chieffs'][v['chieffs']<0.0]))
            maxchis[k].append(v['chieffs'][np.argmax(v['pchieff'][j])])
    macdict = {k: {} for k in below_0.keys()}
    for k,v in macdict.items():
        v['FracBelowNeg0p3'] = save_param_cred_intervals(np.array(below_m0p3[k]))
        v['FracBelow0'] = save_param_cred_intervals(np.array(below_0[k]))
        v['PeakChiEff'] = save_param_cred_intervals(np.array(maxchis[k]))
        v['frac_dyn'] = save_param_cred_intervals(2.0*np.array(below_0[k]))
        v['frac_hm'] = save_param_cred_intervals(6.25*np.array(below_m0p3[k]))
    return macdict

def PLPeakMacros():
    posterior = load_o3b_posterior('o1o2o3_mass_c_iid_mag_iid_tilt_powerlaw_redshift_result.json')
    return {'beta': save_param_cred_intervals(posterior['beta']), 
            'local_rate': save_param_cred_intervals(posterior['rate']),
            'lamb': save_param_cred_intervals(posterior['lamb'])}

def main():
    macro_dict = {}
    macro_dict["PLPeak"] = PLPeakMacros()
    macro_dict["Default"] = DefaultSpinMacros()
    macro_dict["BSplineIndependentCompSpins"] = BSplineIndSpinMacros()
    macro_dict["BSplineIIDCompSpins"] = BSplineIIDSpinMacros()
    macro_dict["ChiEffective"] = chi_eff()
    macro_dict['MassDistribution'] = BSplineMassMacros()
    
    print("Saving macros to src/data/macros.json...")
    with open(paths.data / "macros.json", 'w') as f:
        json.dump(macro_dict, f)
    print("Updating macros in src/tex/macros.tex from data in src/data/macros.json...")
    with open("src/tex/macros.tex", 'w') as ff:
        json2latex.dump('macros', macro_dict, ff)

if __name__ == '__main__':
    main()