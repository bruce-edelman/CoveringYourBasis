#!/usr/bin/env python

import json2latex
import json
import paths
import numpy as np
from utils import load_iid_tilt_ppd, load_ind_tilt_ppd

def save_param_cred_intervals(param_data):
    return  {'median': "{:.2f}".format(np.median(param_data)), 
             'error plus': "{:.2f}".format(np.percentile(param_data, 95)-np.mean(param_data)), 
             'error minus': "{:.2f}".format(np.median(param_data)-np.percentile(param_data, 5)),
             '5th percentile': "{:.2f}".format(np.percentile(param_data, 5)), 
             '95th percentile': "{:.2f}".format(np.percentile(param_data, 95))}

macro_dict = {}
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
macro_dict['MSplineIIDCompSpins'] = {'peakCosTilt': save_param_cred_intervals(peak_tilts), 
                                     'log10gammaFrac': save_param_cred_intervals(np.log10(gamma_fracs)), 
                                     'negFrac': save_param_cred_intervals(frac_neg_cts)}

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

macro_dict['MSplineIndependentCompSpins'] = {'peakCosTilt1': save_param_cred_intervals(peak_tilts1), 'peakCosTilt2': save_param_cred_intervals(peak_tilts2), 
                                             'log10gammaFrac1': save_param_cred_intervals(np.log10(gamma_fracs1)), 'log10gammaFrac2': save_param_cred_intervals(np.log10(gamma_fracs2)), 
                                             'negFrac1': save_param_cred_intervals(frac_neg_cts1), 'negFrac2': save_param_cred_intervals(frac_neg_cts2)}

print("Saving macros to src/data/macros.json...")
with open(paths.data / "macros.json", 'w') as f:
    json.dump(macro_dict, f)
print("Updating macros in src/tex/macros.tex from data in src/data/macros.json...")
with open("src/tex/macros.tex", 'w') as ff:
    json2latex.dump('macros', macro_dict, ff)