#!/usr/bin/env python

import json2latex
import paths
import deepdish as dd
import numpy as np

def load_tilt_ppd():
    datadict = dd.io.load(paths.data / 'mspline_50m1_16iid_compspins_smoothprior_powerlaw_q_z_ppds.h5')
    return datadict['tilts'], datadict['dRdct']

def load_ind_tilt_ppd():
    datadict = dd.io.load(paths.data / 'mspline_50m1_16ind_compspins_smoothprior_powerlaw_q_z_ppds.h5')
    return datadict['tilts'], datadict['dRdct1'], datadict['dRdct2']

def save_param_cred_intervals(param_data):
    return  {'median': "{:.2f}".format(np.median(param_data)), 
             'error plus': "{:.2f}".format(np.percentile(param_data, 95)-np.mean(param_data)), 
             'error minus': "{:.2f}".format(np.median(param_data)-np.percentile(param_data, 5)),
             '5th percentile': "{:.2f}".format(np.percentile(param_data, 5)), 
             '95th percentile': "{:.2f}".format(np.percentile(param_data, 95))}

macro_dict = {}
print("Saving MSpline IID Component Spin macros...")
xs, ct_pdfs = load_tilt_ppd()
peak_tilts = []
for jj in range(len(ct_pdfs)):
    peak_tilts.append(xs[np.argmax(ct_pdfs[jj,:])])
peak_tilts = np.array(peak_tilts)
macro_dict['MSPlineIIDCompSpins'] = {'peakCosTilt': save_param_cred_intervals(peak_tilts)}

print("Saving MSpline Independent Component Spin macros...")
xs, ct1_pdfs, ct2_pdfs = load_ind_tilt_ppd()
peak_tilts1 = []
peak_tilts2 = []
for jj in range(len(ct1_pdfs)):
    peak_tilts1.append(xs[np.argmax(ct1_pdfs[jj,:])])
    peak_tilts2.append(xs[np.argmax(ct2_pdfs[jj,:])])
peak_tilts1 = np.array(peak_tilts1)
peak_tilts2 = np.array(peak_tilts2)
macro_dict['MSPlineIndependentCompSpins'] = {'peakCosTilt1': save_param_cred_intervals(peak_tilts1)}
macro_dict['MSPlineIndependentCompSpins'] = {'peakCosTilt2': save_param_cred_intervals(peak_tilts2)}

print("Updating macros in src/tex/macros.tex from data in src/data/macros.json")
with open("src/tex/macros.tex", 'w') as ff:
    json2latex.dump('macros', macro_dict, ff)