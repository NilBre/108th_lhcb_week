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

files_Tx = [\
         "txt_files/parsed_out/parsedlog_269045_Tx_0_2mm.json",
         "txt_files/parsed_out/parsedlog_269045_Tx_0_4mm.json",
         "txt_files/parsed_out/parsedlog_269045_Tx_0_6mm.json",
         "txt_files/parsed_out/parsedlog_269045_Tx_0_8mm.json",
         "txt_files/parsed_out/parsedlog_269045_base_noSurvey.json",
         "txt_files/parsed_out/Tx/parsedlog_269045_Tx_2mm_noS.json",
         "txt_files/parsed_out/Tx/parsedlog_269045_Tx_3mm_noS.json",
         "txt_files/parsed_out/Tx/parsedlog_269045_Tx_4mm_noS.json",
         "txt_files/parsed_out/Tx/parsedlog_269045_Tx_5mm_noS.json",
         "txt_files/parsed_out/Tx/parsedlog_269045_Tx_6mm_noS.json",
         "txt_files/parsed_out/Tx/parsedlog_269045_Tx_7mm_noS.json",
         "txt_files/parsed_out/Tx/parsedlog_269045_Tx_8mm_noS.json",
         "txt_files/parsed_out/Tx/parsedlog_269045_Tx_9mm_noS.json",
         "txt_files/parsed_out/Tx/parsedlog_269045_Tx_10mm_noS.json",
]
files_Ty = [\
         "txt_files/parsed_out/parsedlog_269045_base_noSurvey.json",
         "txt_files/parsed_out/Ty/parsedlog_269045_Ty_2mm_noS.json",
         "txt_files/parsed_out/Ty/parsedlog_269045_Ty_3mm_noS.json",
         "txt_files/parsed_out/Ty/parsedlog_269045_Ty_4mm_noS.json",
         "txt_files/parsed_out/Ty/parsedlog_269045_Ty_5mm_noS.json",
         "txt_files/parsed_out/Ty/parsedlog_269045_Ty_6mm_noS.json",
         "txt_files/parsed_out/Ty/parsedlog_269045_Ty_7mm_noS.json",
         "txt_files/parsed_out/Ty/parsedlog_269045_Ty_8mm_noS.json",
         "txt_files/parsed_out/Ty/parsedlog_269045_Ty_9mm_noS.json",
         "txt_files/parsed_out/Ty/parsedlog_269045_Ty_10mm_noS.json",
]
files_Rx = [\
        "txt_files/parsed_out/parsedlog_269045_base_noSurvey.json",
        "txt_files/parsed_out/Rx/parsedlog_Rx_03mrad.json",
        "txt_files/parsed_out/Rx/parsedlog_Rx_04mrad.json",
        "txt_files/parsed_out/Rx/parsedlog_Rx_05mrad.json"
]
files_Ry = [\
        "txt_files/parsed_out/Ry/parsedlog_Ry_0_4mrad.json", # 0.0004
        "txt_files/parsed_out/Ry/parsedlog_Ry_0_3mrad.json", # 0.0003
        "txt_files/parsed_out/parsedlog_269045_base_noSurvey.json", # 0.0002 base
        "txt_files/parsed_out/Ry/parsedlog_Ry_0_1mrad.json", # 0.0001
        "txt_files/parsed_out/Ry/parsedlog_Ry_0_02mrad.json", # 0.00002
        "txt_files/parsed_out/Ry/parsedlog_Ry_3microrad.json",
        "txt_files/parsed_out/Ry/parsedlog_Ry_2microrad.json",
        "txt_files/parsed_out/Ry/parsedlog_Ry_1microrad.json",
        "txt_files/parsed_out/Ry/parsedlog_Ry_0_8microrad.json",
        "txt_files/parsed_out/Ry/parsedlog_Ry_0_6microrad.json",
        "txt_files/parsed_out/Ry/parsedlog_Ry_0_5microrad.json",
        "txt_files/parsed_out/Ry/parsedlog_Ry_0_4microrad.json",
        "txt_files/parsed_out/Ry/parsedlog_Ry_0_44microrad.json",
        "txt_files/parsed_out/Ry/parsedlog_Ry_0_2microrad.json",
]

legendlabels_Tx=[\
        "0_2mm",
        "0_4mm",
        "0_6mm",
        "0_8mm",
        "base",
        "2mm",
        "3mm",
        "4mm",
        "5mm",
        "6mm",
        "7mm",
        "8mm",
        "9mm",
        "0mm"
]
legendlabels_Ty = [\
        "base",
        "2mm",
        "3mm",
        "4mm",
        "5mm",
        "6mm",
        "7mm",
        "8mm",
        "9mm",
        "10mm",
]
legendlabels_Rx = [\
        "base",
        "03mrad",
        "04mrad",
        "05mrad",
]

legendlabels_Ry = [\
        "04mrad",
        "03mrad",
        "base",
        "01mrad",
        "0_02mrad",
        "3microRad",
        "2microRad",
        "1microRad",
        "08microRad",
        "06microRad",
        "05microRad",
        "0_44microRad",
        "04microRad",
        "02microRad",
]

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

chi2_values_from_Tx_changes = get_chi2_values(files_Tx)
chi2_values_from_Ty_changes = get_chi2_values(files_Ty)
chi2_values_from_Rx_changes = get_chi2_values(files_Rx)
chi2_values_from_Ry_changes = get_chi2_values(files_Ry)

Tz_chi2_vals = [2586.76]

# 7mm and 8mm are missing, last entries are 5mm, 6mm, 7mm, 10mm
Tx_chi2_tz = [15692/768, 8182.69/768, 5278.57/768, 5770.66/768, 12196.7/768, 9622.9/768, 10590.6/768]
Ty_chi2_tz = [1127.64/768, 1122.31/768, 1119.17/768, 1120.25/768, 1125.93/768, 1125.58/768, 1126.06/768]
Tz_chi2_tz = [646.69/768, 287.418/768, 161.672/768, 103.47/768, 71.8544/768, 52.791/768, 25.8676/768]
Rx_chi2_tz = [3086.38/768, 3086.38/768, 3086.38/768, 3086.38/768, 3086.38/768, 3086.38/768, 3086.38/768]
Ry_chi2_tz = [0.00366032/768, 0.0036605/768, 0.00369746/768, 0.00370069/768, 0.0036554/768, 0.00368278/768, 0.0036503/768]
Rz_chi2_tz = [708.737/768, 679.203/768, 658.385/768, 658.059/768, 697.696/768, 678.97/768,  687.408/768]

Tz_data = np.array([Tx_chi2_tz, Ty_chi2_tz, Tz_chi2_tz, Rx_chi2_tz, Ry_chi2_tz, Rz_chi2_tz])
print(Tz_data)
correct_order = [2, 0, 1, 3, 6, 4, 5, 7, 10, 8, 9, 11]
Tx_err = [0.2, 0.4, 0.6, 0.8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # microns
Ty_err = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] # microns
Tz_err = [2,3,4,5,6,7,10] # microns

Rx_err = [0.2, 0.3, 0.4, 0.5] # microrad
Ry_err = [0.4, 0.3, 0.2, 0.1, 0.02, 0.003, 0.002, 0.001, 0.0008, 0.0006, 0.0005, 0.00044, 0.0004, 0.0002] # microrad

total_n_dofs = 768

def plotting(x_range, data, dofs, label):
    if label == 'Ry':
        x_vals = np.linspace(0, 14, 14)
        print(f"label: {label}")
        print(data[0])
        plt.plot(x_vals, data[0] / dofs, marker='.', linestyle='--', label='Tx_chi')
        plt.plot(x_vals, data[1] / dofs, marker='.', linestyle='--', label='Ty_chi')
        plt.plot(x_vals, data[2] / dofs, marker='.', linestyle='--', label='Tz_chi')
        plt.plot(x_vals, data[3] / dofs, marker='.', linestyle='--', label='Rx_chi')
        plt.plot(x_vals, data[4] / dofs, marker='.', linestyle='--', label='Ry_chi')
        plt.plot(x_vals, data[5] / dofs, marker='.', linestyle='--', label='Rz_chi')
        plt.xticks(x_vals, x_range, rotation=45)
        plt.xlabel('[mu rad]')
    if label == 'Tz':
        print(f"label: {label}")
        print(data[0])
        plt.plot(x_range, data[0], marker='.', linestyle='--', label='Tx_chi')
        plt.plot(x_range, data[1], marker='.', linestyle='--', label='Ty_chi')
        plt.plot(x_range, data[2], marker='.', linestyle='--', label='Tz_chi')
        plt.plot(x_range, data[3], marker='.', linestyle='--', label='Rx_chi')
        plt.plot(x_range, data[4], marker='.', linestyle='--', label='Ry_chi')
        plt.plot(x_range, data[5], marker='.', linestyle='--', label='Rz_chi')
    if label in ['Tx', 'Ty']:
        print(f'else: {label}')
        print(data)
        plt.plot(x_range, data[0] / dofs, marker='.', linestyle='--', label='Tx_chi')
        plt.plot(x_range, data[1] / dofs, marker='.', linestyle='--', label='Ty_chi')
        plt.plot(x_range, data[2] / dofs, marker='.', linestyle='--', label='Tz_chi')
        plt.plot(x_range, data[3] / dofs, marker='.', linestyle='--', label='Rx_chi')
        plt.plot(x_range, data[4] / dofs, marker='.', linestyle='--', label='Ry_chi')
        plt.plot(x_range, data[5] / dofs, marker='.', linestyle='--', label='Rz_chi')
        plt.xlabel('[mu m]')
    if label == 'Rx':
        print(f'else: {label}')
        print(data)
        plt.plot(x_range, data[0] / dofs, marker='.', linestyle='--', label='Tx_chi')
        plt.plot(x_range, data[1] / dofs, marker='.', linestyle='--', label='Ty_chi')
        plt.plot(x_range, data[2] / dofs, marker='.', linestyle='--', label='Tz_chi')
        plt.plot(x_range, data[3] / dofs, marker='.', linestyle='--', label='Rx_chi')
        plt.plot(x_range, data[4] / dofs, marker='.', linestyle='--', label='Ry_chi')
        plt.plot(x_range, data[5] / dofs, marker='.', linestyle='--', label='Rz_chi')
        plt.xlabel('[mu rad]')
    plt.hlines(1, min(x_range), max(x_range), 'black')
    plt.legend()
    plt.grid()
    plt.title(f'chi2 / dof changing only {label} error')
    plt.ylabel('chi2 per dof')
    # plt.show()
    plt.savefig(f'chi2_plots/{label}_out.pdf')
    plt.clf()

plotting(Tx_err, chi2_values_from_Tx_changes, total_n_dofs, 'Tx')
plotting(Ty_err, chi2_values_from_Ty_changes, total_n_dofs, 'Ty')
plotting(Tz_err, Tz_data, total_n_dofs, 'Tz')
plotting(Rx_err, chi2_values_from_Rx_changes, total_n_dofs, 'Rx')
plotting(Ry_err, chi2_values_from_Ry_changes, total_n_dofs, 'Ry')

errors_v1 = [0.001, 0.0015, 0.004, 0.0002, 0.0002, 0.0002]
errors_v2 = [0.0005, 0.002, 0.002, 0.0002, 0.0002, 0.0002]
errors_v3 = [0.0002, 0.002, 0.002, 0.0002, 0.0002, 0.0002]
errors_v4 = [0.00018, 0.0015, 0.0018, 0.0004, 0.00000044, 0.00019]

from scipy.stats import linregress
x1 = Tx_err[0:7]
y1 = (chi2_values_from_Tx_changes[0][0:7] / total_n_dofs).T[0]
print(x1, y1)
erg = linregress(x1, y1)
print(erg)

# try curve fit
from scipy.optimize import curve_fit
def func(x, a, b, c):
    return a * np.exp(-b * x) + c
def Gauss(x, A, B):
    y = A*np.exp(-1*B*x**2)
    return y
x2 = Tx_err[6:]
y2 = (chi2_values_from_Tx_changes[0][6:] / total_n_dofs).T[0]

xfull = np.array(Tx_err)
yfull = np.array((chi2_values_from_Tx_changes[0] / total_n_dofs).T[0])

xdata1 = np.array(x1)
ydata1 = np.array(y1)

xdata2 = np.array(x2)
ydata2 = np.array(y2)

plt.plot(xfull, yfull, 'k-', label='data')
popt1, pcov1 = curve_fit(func, xdata1, ydata1)
popt2, pcov2 = curve_fit(func, xdata2, ydata2)
plt.plot(xdata1, func(xdata1, *popt1), 'r-',label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt1))
# plt.plot(xdata2, func(xdata2, *popt2), 'b-',label='fit: A=%5.3f, B=%5.3f, c=%5.3f' % tuple(popt2))
plt.axvline(x = 0.29, color = 'g', linestyle='dashed', label='Tx unc. = 0.29 micron')
plt.axhline(y = 1, color = 'y', linestyle='dashed', label='chi2 / dof = 1')
plt.grid()
plt.legend()
plt.xlabel('Tx unc [micron]')
plt.ylabel('chi2 / dof')
plt.title('Tx uncertainty tuning')
plt.savefig('fit_Tx.pdf')
plt.show()
# print(func(xdata, *popt))

# FIXME:
# 1. correct the x label to be LaTeX formated
# 2. Tx: more alignments between 0 and 1 microns, also fit a function (linear, polynomial 2nd order and find chi2 / dof = 1)
# 3. Ty: more alignments between 1.2 and 1.7 microns
# 4. Tz: more alignments between 1 and 2 microns
# 5. Rx: more alignments around 0.4 micro rad (0.0004 milli rad)
# 6. Ry: more alignments roughly at 0.00044 microns
# 7: Rz: please also do a plot for Rz around 0.0002 milli rad
