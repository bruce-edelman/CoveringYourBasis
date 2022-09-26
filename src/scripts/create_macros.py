#!/usr/bin/env python

import json2latex
import json
import paths
import numpy as np
import deepdish as dd
from scipy.integrate import cumtrapz
from utils import load_iid_tilt_ppd, load_ind_tilt_ppd, save_param_cred_intervals, load_o3b_paper_run_masspdf, load_mass_ppd


def get_m1_m99(pdfs, ms):
    m99 = []
    m1 = []
    for m in pdfs:                                                                                                                                                                        
        i = len(m)
        cumulative_prob = cumtrapz(m, ms, initial = 0)
        init_prob = cumulative_prob[-1]
        prob = init_prob
        final_prob = init_prob*0.99                                                                                                                                              
        while prob > (final_prob):
            i -= 1
            prob = cumulative_prob[i]                                                                                                                                                         
        m99.append(ms[i])
    m99 = np.array(m99)
    for m in pdfs:
        i = 0
        cumulative_prob = cumtrapz(m, ms, initial = 0)
        init_prob = cumulative_prob[-1]
        prob = 0
        final_prob = init_prob*0.01
        while prob < (final_prob):
            i += 1
            prob = cumulative_prob[i]
        m1.append(ms[i])
    m1 = np.array(m1)
    return m1, m99


def MSplineMassMacros():
    print("Saving Mass Distribution Macros...")
    ms, m_pdfs, _, _ = load_mass_ppd()
    m1s, m99s = get_m1_m99(m_pdfs, ms)
    plpeak_mpdfs, _, plpeak_ms, _ = load_o3b_paper_run_masspdf(paths.data / 'o1o2o3_mass_c_iid_mag_iid_tilt_powerlaw_redshift_mass_data.h5')
    plpeak_m1s, plpeak_m99s = get_m1_m99(plpeak_mpdfs, plpeak_ms)
    ps_mpdfs, _, ps_ms, _ = load_o3b_paper_run_masspdf(paths.data / 'spline_20n_mass_m_iid_mag_iid_tilt_powerlaw_redshift_mass_data.h5')
    ps_m1s, ps_m99s = get_m1_m99(ps_mpdfs, ps_ms)
    return {'PLPeak': {'m_1percentile': save_param_cred_intervals(plpeak_m1s), 'm_99percentile': save_param_cred_intervals(plpeak_m99s)}, 
            'MSpline': {'m_1percentile': save_param_cred_intervals(m1s), 'm_99percentile': save_param_cred_intervals(m99s)}, 
            'PLSpline': {'m_1percentile': save_param_cred_intervals(ps_m1s), 'm_99percentile': save_param_cred_intervals(ps_m99s)}}

def MSplineIIDSpinMacros():
    print("Saving MSpline IID Component Spin macros...")
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
    return {'peakCosTilt': save_param_cred_intervals(peak_tilts), 
            'log10gammaFrac': save_param_cred_intervals(np.log10(gamma_fracs)), 
            'negFrac': save_param_cred_intervals(frac_neg_cts)}


def MSplineIndSpinMacros():
    print("Saving MSpline Independent Component Spin macros...")
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

    return {'peakCosTilt1': save_param_cred_intervals(peak_tilts1), 'peakCosTilt2': save_param_cred_intervals(peak_tilts2), 
            'log10gammaFrac1': save_param_cred_intervals(np.log10(gamma_fracs1)), 'log10gammaFrac2': save_param_cred_intervals(np.log10(gamma_fracs2)), 
            'negFrac1': save_param_cred_intervals(frac_neg_cts1), 'negFrac2': save_param_cred_intervals(frac_neg_cts2)}
    
    
def chi_eff():
    print("Saving ChiEffective macros...")
    ppds = dd.io.load(paths.data / "chi_eff_ppds.h5")
    default = ppds["Default"]
    msplind = ppds["MSplineInd"]
    mspliid = ppds['MSplineIID']
    msplchieff = dd.io.load(paths.data / "mspline_50m1_24chieff_smoothprior_powerlaw_q_z_fitlamb_ppds.h5")
    below_0 = {'default': [], 'iid': [], 'ind': [], 'chieff': []}
    below_m0p3 = {'default': [], 'iid': [], 'ind': [], 'chieff': []}
    maxchis = {'default': [], 'iid': [], 'ind': [], 'chieff': []}
    for i in range(1500):
        norm = np.trapz(msplchieff['dRdchieff'][i, :], x=msplchieff['chieffs'])
        below_m0p3['chieff'].append(np.trapz(1./norm*msplchieff['dRdchieff'][i, msplchieff['chieffs']<-0.3], x=msplchieff['chieffs'][msplchieff['chieffs']<-0.3]))
        below_0['chieff'].append(np.trapz(1./norm*msplchieff['dRdchieff'][i, msplchieff['chieffs']<0.0], x=msplchieff['chieffs'][msplchieff['chieffs']<0.0]))
        maxchis['chieff'].append(msplchieff['chieffs'][np.argmax(msplchieff['dRdchieff'][i])])
    for j in range(500):
        for k,v in zip(['default', 'ind', 'iid'], [default, msplind, mspliid]):
            below_m0p3[k].append(np.trapz(v['pchieff'][j,v['chieffs']<-0.3], x=v['chieffs'][v['chieffs']<-0.3]))
            below_0[k].append(np.trapz(v['pchieff'][j,v['chieffs']<0.0], x=v['chieffs'][v['chieffs']<0.0]))
            maxchis[k].append(v['chieffs'][np.argmax(v['pchieff'][j])])
    macdict = {k: {} for k in below_0.keys()}
    for k,v in macdict.items():
        v['FracBelowNeg0p3'] = save_param_cred_intervals(np.array(below_m0p3[k]))
        v['FracBelow0'] = save_param_cred_intervals(np.array(below_0[k]))
        v['PeakChiEff'] = save_param_cred_intervals(np.array(maxchis[k]))
    return macdict

def main():
    macro_dict = {}
    macro_dict["MSplineIndependentCompSpins"] = MSplineIndSpinMacros()
    macro_dict["MSplineIIDCompSpins"] = MSplineIIDSpinMacros()
    macro_dict["ChiEffective"] = chi_eff()
    macro_dict['MassDistribution'] = MSplineMassMacros()
    
    print("Saving macros to src/data/macros.json...")
    with open(paths.data / "macros.json", 'w') as f:
        json.dump(macro_dict, f)
    print("Updating macros in src/tex/macros.tex from data in src/data/macros.json...")
    with open("src/tex/macros.tex", 'w') as ff:
        json2latex.dump('macros', macro_dict, ff)

if __name__ == '__main__':
    main()