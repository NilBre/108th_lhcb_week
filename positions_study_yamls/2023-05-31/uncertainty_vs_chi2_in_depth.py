import numpy as np
import matplotlib.pyplot as plt
from math import *
# import ROOT
import matplotlib.patches as mpatches
import statistics
from copy import deepcopy
import sys
import os
import random
import yaml
import matplotlib
import glob
from grepfunc import grep_iter
import re
import json
import argparse
from termcolor import colored
import mplhep as hep
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import AxesGrid
from scipy.stats import sem
from scipy.optimize import curve_fit

plt.rcParams['text.usetex'] = True

regex_typelabel=re.compile("Q")
regex_amodule=re.compile("dPosXYZ")
regex_rmodule=re.compile("dRotXYZ")
labels=["Tx","Ty","Tz","Rx","Ry","Rz"]
positions=["x_global","y_global","z_global"]
trackInfo=["nTracks","nHits"]
stations = ["T1", "T2", "T3"]
layers = ["U", "V", "X1", "X2"]

colors = ['black', 'blue', 'red', 'green', 'yellow', 'magenta', 'brown', 'cyan']
markers = ['o', 'x', 'd', 'D', '.', 'v', 's', 'p']

def func(x, a, b, c):
    return a * np.exp(-b * x) + c

def f_tanh(x, eta=1, phi=0):
    "tanh function"
    return np.tanh(eta * (x + phi))

def x_intersection(a, b, c):
    return (-1 / b) * (np.log(1 - c) - np.log(a))

def open_alignment(thisfile,convergence=True):
    with open(thisfile) as f:
        align_output=json.load(f)

    convergences=align_output.pop("converged")

    for alignable in align_output.keys():
        for label in labels+positions+trackInfo:
            if "FT" in alignable:
                align_output[alignable][label]=[float(ele.strip(',')) for ele in align_output[alignable][label]]

    if convergence:
        align_output["convergence"]=convergences
    return align_output

def convertGlobal(align_external,halfmoduleAlignables):
    align_output=deepcopy(align_external)
    for alignable in halfmoduleAlignables:
        # rule for readout flip in x per layer:
        if "Quarter0" in alignable or "Quarter1" in alignable or "Q0" in alignable or "Q1" in alignable:
            for label in ["Tx","Rx","Ry","Ty"]:
                if label in align_output[alignable].keys():
                    align_output[alignable][label]=[-value for value in align_output[alignable][label]]
            # minus sign on x and Rx and y and Ry for all iterations
        if "X1" in alignable or "V" in alignable:
            # minus sign on x and Rx and z and Rz for all iterations
            for label in ["Tx","Rx","Rz","Tz"]:
                if label in align_output[alignable].keys():
                    align_output[alignable][label]=[-value for value in align_output[alignable][label]]

    return align_output

def get_unc(input):
    n_files = len(input)
    Tx_chi2 = [[] for _ in range(n_files)]
    Ty_chi2 = [[] for _ in range(n_files)]
    Tz_chi2 = [[] for _ in range(n_files)]

    Rx_chi2 = [[] for _ in range(n_files)]
    Ry_chi2 = [[] for _ in range(n_files)]
    Rz_chi2 = [[] for _ in range(n_files)]
    for i in range(n_files):
        Tx_chi2[i].append(input[i]['Tx_chi2'][0])
        Ty_chi2[i].append(input[i]['Ty_chi2'][0])
        Tz_chi2[i].append(input[i]['Tz_chi2'][0])

        Rx_chi2[i].append(input[i]['Rx_chi2'][0])
        Ry_chi2[i].append(input[i]['Ry_chi2'][0])
        Rz_chi2[i].append(input[i]['Rz_chi2'][0])

    return np.array(Tx_chi2), np.array(Ty_chi2), np.array(Tz_chi2), np.array(Rx_chi2), np.array(Ry_chi2), np.array(Rz_chi2)

def get_chi2_values(input_files):
    outputs=[open_alignment(thisfile) for thisfile in input_files]
    plotted_alignables=[]
    for align_block in outputs:
        thislist=[]
        for key in align_block.keys():
            thislist.append(key)
        plotted_alignables.append(thislist)
    outputs=[convertGlobal(align_block,plotted_alignables[0]) for align_block in outputs]

    Tx_chi2_tx = get_unc(outputs)[0]
    Ty_chi2_tx = get_unc(outputs)[1]
    Tz_chi2_tx = get_unc(outputs)[2]
    Rx_chi2_tx = get_unc(outputs)[3]
    Ry_chi2_tx = get_unc(outputs)[4]
    Rz_chi2_tx = get_unc(outputs)[5]

    return Tx_chi2_tx, Ty_chi2_tx, Tz_chi2_tx, Rx_chi2_tx, Ry_chi2_tx, Rz_chi2_tx

def plotting(x_range, data, dofs, label, name):
    range = len(x_range)
    x_vals = np.linspace(0, range, range)
    # print(f'########## data for {label} ##########')
    # print('data:', data)
    # print('data[0]:', data[0])
    index = 0
    if label in ['Tx', 'Ty', 'Tz']:
        if label == 'Ty':
            index = 1
        if label == 'Tz':
            index = 2
        x = x_range[0:range-1] # excldue last value
        y = (data[index][0:range-1] / dofs).T[0]
        # x = x_range
        # y = (data[index] / dofs).T[0]
        xdata = np.array(x)
        ydata = np.array(y)
        popt, pcov = curve_fit(func, xdata, ydata)
        plt.plot(xdata, func(xdata, *popt), 'r-',label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
        plt.plot(x_range, data[0] / dofs, marker='.', linestyle='--', label='Tx_chi')
        plt.plot(x_range, data[1] / dofs, marker='.', linestyle='--', label='Ty_chi')
        plt.plot(x_range, data[2] / dofs, marker='.', linestyle='--', label='Tz_chi')
        plt.plot(x_range, data[3] / dofs, marker='.', linestyle='--', label='Rx_chi')
        plt.plot(x_range, data[4] / dofs, marker='.', linestyle='--', label='Ry_chi')
        plt.plot(x_range, data[5] / dofs, marker='.', linestyle='--', label='Rz_chi')
        # plt.xticks(x_vals, x_range, rotation=45)
        plt.xlabel('[micron]')
    if label in ['Rx', 'Ry', 'Rz']:
        if label == 'Rx':
            index = 3
        if label == 'Ry':
            index = 4
        if label == 'Rz':
            index = 5
        x = x_range[0:range-1] # excldue last value
        y = (data[index][0:range-1] / dofs).T[0]
        # x = x_range
        # y = (data[index] / dofs).T[0]
        xdata = np.array(x)
        ydata = np.array(y)
        popt, pcov = curve_fit(func, xdata, ydata)
        plt.plot(xdata, func(xdata, *popt), 'r-',label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
        plt.plot(x_range, data[0] / dofs, marker='.', linestyle='--', label='Tx_chi')
        plt.plot(x_range, data[1] / dofs, marker='.', linestyle='--', label='Ty_chi')
        plt.plot(x_range, data[2] / dofs, marker='.', linestyle='--', label='Tz_chi')
        plt.plot(x_range, data[3] / dofs, marker='.', linestyle='--', label='Rx_chi')
        plt.plot(x_range, data[4] / dofs, marker='.', linestyle='--', label='Ry_chi')
        plt.plot(x_range, data[5] / dofs, marker='.', linestyle='--', label='Rz_chi')
        if label == 'Ry':
            plt.xlabel(r'$[\mu rad]$')
        else:
            plt.xlabel('[mrad]')
    plt.hlines(1, min(x_range), max(x_range), 'black')
    # plt.hlines(1, min(x_range), max(x_range), 'black')
    plt.legend()
    plt.grid()
    plt.title(f'chi2 per dof changing only {label} uncertainty')
    plt.ylabel('chi2 / dof')
    # plt.show()
    plt.savefig(f'chi2_plots/retest/{name}_exclude_last_val.pdf')
    # plt.savefig(f'chi2_plots/retest/{name}_full_fit.pdf')
    plt.clf()
    a = popt[0]
    b = popt[1]
    c = popt[2]
    x_intersect = x_intersection(a, b, c) # in different coords
    if label in ['Tx', 'Ty', 'Tz']:
        print(f'for {label}: intersection for chi2 / dof = 1:', x_intersect, 'micron')
    else:
        print(f'for {label}: intersection for chi2 / dof = 1:', x_intersect, 'mrad')
# Tx
files_Tx = [\
        "retest_uncertainty/json/Tx/parsedlog_Tx_0_1_micron.json",
        "retest_uncertainty/json/Tx/parsedlog_Tx_0_15_micron.json",
        "retest_uncertainty/json/Tx/parsedlog_Tx_0_2_micron.json",
        "retest_uncertainty/json/Tx/parsedlog_Tx_0_25_micron.json",
        "retest_uncertainty/json/Tx/parsedlog_Tx_0_3_micron.json",
        "retest_uncertainty/json/Tx/parsedlog_Tx_0_3001_micron.json",
        "retest_uncertainty/json/Tx/parsedlog_Tx_0_301_micron.json",
        "retest_uncertainty/json/Tx/parsedlog_Tx_0_31_micron.json",
        "retest_uncertainty/json/Tx/parsedlog_Tx_0_35_micron.json",
        "retest_uncertainty/json/Tx/parsedlog_Tx_0_4_micron.json",
        "retest_uncertainty/json/Tx/parsedlog_Tx_0_45_micron.json",
        "retest_uncertainty/json/Tx/parsedlog_Tx_0_5_micron.json",
        "retest_uncertainty/json/parsedlog_base.json",
]
legendlabels_Tx=[\
        "0_1",
        "0_15",
        "0_2",
        "0_25",
        "0_3",
        "0_3001",
        "0_301",
        "0_31",
        "0_35",
        "0_4",
        "0_45",
        "0_5",
        "base",
]

# Ty, all these have Tx = 0.3 micron since this was the best from Tx
files_Ty = [\
         "retest_uncertainty/json/Ty/parsedlog_Ty_4_micron.json",
         "retest_uncertainty/json/Ty/parsedlog_Ty_3_micron.json",
         "retest_uncertainty/json/Ty/parsedlog_Ty_2_micron.json",
         "retest_uncertainty/json/Ty/parsedlog_Ty_1_5_micron.json",
         "retest_uncertainty/json/Ty/parsedlog_Ty_1_4_micron.json",
         "retest_uncertainty/json/Ty/parsedlog_Ty_1_3_micron.json",
         "retest_uncertainty/json/Ty/parsedlog_Ty_1_2_micron.json",
         "retest_uncertainty/json/Ty/parsedlog_Ty_1_1_micron.json",
         "retest_uncertainty/json/Tx/parsedlog_Tx_0_3_micron.json",
]
legendlabels_Ty = [\
    "4",
    "3",
    "2",
    "1_5",
    "1_4",
    "1_3",
    "1_2",
    "1_1",
    "1",
]

# Tz
files_Tz = [\
    "retest_uncertainty/json/Rx/parsedlog_Rx_0_4_mrad.json", # has correct Tx, Tz and Rx for optimal chi2 performance at Tz = 1 micron
    "retest_uncertainty/json/Tz/parsedlog_Tz_1_5_micron.json",
    "retest_uncertainty/json/Tz/parsedlog_Tz_1_7_micron.json",
    "retest_uncertainty/json/Tz/parsedlog_Tz_1_8_micron.json",
    "retest_uncertainty/json/Tz/parsedlog_Tz_1_9_micron.json",
    "retest_uncertainty/json/Tz/parsedlog_Tz_2_micron.json",
]
legendlabels_Tz = [\
    "1",
    "1_5",
    "1_7",
    "1_8",
    "1_9",
    "2",
]

# Rx
files_Rx = [\
        "retest_uncertainty/json/Ty/parsedlog_Ty_1_2_micron.json",
        "retest_uncertainty/json/Rx/parsedlog_Rx_0_3_mrad.json",
        "retest_uncertainty/json/Rx/parsedlog_Rx_0_37_mrad.json",
        "retest_uncertainty/json/Rx/parsedlog_Rx_0_4_mrad.json",
        "retest_uncertainty/json/Rx/parsedlog_Rx_0_5_mrad.json",
]
legendlabels_Rx = [\
    "0_2",
    "0_3",
    "0_37",
    "0_4",
    "0_5",
]

files_Rx_onlyTx = [\
        "retest_uncertainty/json/Tx/parsedlog_Tx_0_3_micron.json",
        "retest_uncertainty/json/Rx/parsedlog_OT_Rx_0_3_mrad.json",
        "retest_uncertainty/json/Rx/parsedlog_OT_Rx_0_4_mrad.json",
        "retest_uncertainty/json/Rx/parsedlog_OT_Rx_0_5_mrad.json",
]
legendlabels_Rx_onlyTx = [\
    "0_2",
    "0_3",
    "0_4",
    "0_5",
]

# Ry
files_Ry = [\
        "retest_uncertainty/json/Ry/parsedlog_Ry_0_3_micro_rad.json",
        "retest_uncertainty/json/Ry/parsedlog_Ry_0_35_micro_rad.json",
        "retest_uncertainty/json/Ry/parsedlog_Ry_0_4_micro_rad.json",
        "retest_uncertainty/json/Ry/parsedlog_Ry_0_44_micro_rad.json",
        "retest_uncertainty/json/Ry/parsedlog_Ry_0_5_micro_rad.json",
        "retest_uncertainty/json/Ry/parsedlog_Ry_0_6_micro_rad.json",
        # "retest_uncertainty/json/Tz/parsedlog_Tz_1_8_micron.json",
]
legendlabels_Ry = [\
    "0_3_micro",
    "0_35_micro",
    "0_4_micro",
    "0_44_micro",
    "0_5_micro",
    "0_6_micro",
    # "0_2_mrad",
]

# Rz
files_Rz = [\
    "retest_uncertainty/json/Rz/parsedlog_Rz_0_1_mrad.json",
    "retest_uncertainty/json/Rz/parsedlog_Rz_0_15_mrad.json",
    "retest_uncertainty/json/Ry/parsedlog_Ry_0_44_micro_rad.json",
    "retest_uncertainty/json/Rz/parsedlog_Rz_0_25_mrad.json",
    "retest_uncertainty/json/Rz/parsedlog_Rz_0_3_mrad.json",
    "retest_uncertainty/json/Rz/parsedlog_Rz_0_4_mrad.json",
]
legendlabels_Rz = [\
    "0_1",
    "0_15",
    "0_2",
    "0_25",
    "0_3",
    "0_4",
]

chi2_values_from_Tx_changes = get_chi2_values(files_Tx)
chi2_values_from_Ty_changes = get_chi2_values(files_Ty)
chi2_values_from_Tz_changes = get_chi2_values(files_Tz)
chi2_values_from_Rx_changes = get_chi2_values(files_Rx)
chi2_values_from_Rx_changes_OT = get_chi2_values(files_Rx_onlyTx)
chi2_values_from_Ry_changes = get_chi2_values(files_Ry)
chi2_values_from_Rz_changes = get_chi2_values(files_Rz)

correct_order = [2, 0, 1, 3, 6, 4, 5, 7, 10, 8, 9, 11]
# , 0.3001, 0.301, 0.31
Tx_err = [0.1, 0.15, 0.2, 0.25, 0.3, 0.3001, 0.301, 0.31, 0.35, 0.4, 0.45, 0.5, 1.0]  # mm
Ty_err = [4, 3, 2, 1.5, 1.4, 1.3, 1.2, 1.1, 1] # micron
Tz_err = [1, 1.5, 1.7, 1.8, 1.9, 2] # mm
# that means 0.1 micron fuer Tx first value
Rx_err = [0.2, 0.3, 0.37, 0.4, 0.5] # mrad
Rx_OT_err = [0.2, 0.3, 0.4, 0.5] # mrad
Ry_err = [0.3, 0.35, 0.4, 0.44, 0.5, 0.6] # micro rad
Rz_err = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4] # mrad

# Tx_unc = np.linspace(0.1, 1, 10) # micron, 1 micron is base config
# Ty_unc = np.linspace(1.2, 1.7, 6) # micron
# Tz_unc = np.linspace(1, 2, 6) # 5 additional alignments, 1 micron again is base
# Rx_unc = np.linspace(0.35, 0.45, 11) # mrad
# Ry_unc = np.linspace(0.0003, 0.0005, 6) # mrad
# Rz_unc = np.linspace(0.1, 0.3, 6)

total_n_dofs = 768

plotting(Tx_err, chi2_values_from_Tx_changes, total_n_dofs, 'Tx', 'only_Tx')
plotting(Ty_err, chi2_values_from_Ty_changes, total_n_dofs, 'Ty', 'Ty_with_set_Tx')
plotting(Tz_err, chi2_values_from_Tz_changes, total_n_dofs, 'Tz', 'Tz_set_TxTzRx')
plotting(Rx_err, chi2_values_from_Rx_changes, total_n_dofs, 'Rx', 'Rx_with_set_TxTy')
plotting(Rx_OT_err, chi2_values_from_Rx_changes_OT, total_n_dofs, 'Rx', 'Rx_only_set_Tx')
plotting(Ry_err, chi2_values_from_Ry_changes, total_n_dofs, 'Ry', 'everything_set_but_Ry')
plotting(Tz_err, chi2_values_from_Rz_changes, total_n_dofs, 'Rz', 'only_Rz_variable')

best_Tx = 0.3 # 0.22 fit not working so well
best_Ty = 1.2
best_Tz = 1.9 # 1.83
best_Rx = 0.0004
best_Ry = 0.00000044
best_Rz = 0.0002
