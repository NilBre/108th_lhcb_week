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

    # runs_T1 = ["FT/T1/U/HL0/Q0/M0", "FT/T1/U/HL0/Q0/M1", "FT/T1/U/HL0/Q0/M2", "FT/T1/U/HL0/Q0/M3", "FT/T1/U/HL0/Q0/M4",
    #            "FT/T1/U/HL0/Q2/M0", "FT/T1/U/HL0/Q2/M1", "FT/T1/U/HL0/Q2/M2", "FT/T1/U/HL0/Q2/M3", "FT/T1/U/HL0/Q2/M4",
    #            "FT/T1/U/HL1/Q1/M0", "FT/T1/U/HL1/Q1/M1", "FT/T1/U/HL1/Q1/M2", "FT/T1/U/HL1/Q1/M3", "FT/T1/U/HL1/Q1/M4",
    #            "FT/T1/U/HL1/Q3/M0", "FT/T1/U/HL1/Q3/M1", "FT/T1/U/HL1/Q3/M2", "FT/T1/U/HL1/Q3/M3", "FT/T1/U/HL1/Q3/M4"]
    #
    # runs_T2 = ["FT/T2/U/HL0/Q0/M0", "FT/T2/U/HL0/Q0/M1", "FT/T2/U/HL0/Q0/M2", "FT/T2/U/HL0/Q0/M3", "FT/T2/U/HL0/Q0/M4",
    #            "FT/T2/U/HL0/Q2/M0", "FT/T2/U/HL0/Q2/M1", "FT/T2/U/HL0/Q2/M2", "FT/T2/U/HL0/Q2/M3", "FT/T2/U/HL0/Q2/M4",
    #            "FT/T2/U/HL1/Q1/M0", "FT/T2/U/HL1/Q1/M1", "FT/T2/U/HL1/Q1/M2", "FT/T2/U/HL1/Q1/M3", "FT/T2/U/HL1/Q1/M4",
    #            "FT/T2/U/HL1/Q3/M0", "FT/T2/U/HL1/Q3/M1", "FT/T2/U/HL1/Q3/M2", "FT/T2/U/HL1/Q3/M3", "FT/T2/U/HL1/Q3/M4"]
    #
    # runs = ["FT/T3/U/HL0/Q0/M0", "FT/T3/U/HL0/Q0/M1", "FT/T3/U/HL0/Q0/M2", "FT/T3/U/HL0/Q0/M3", "FT/T3/U/HL0/Q0/M4",
    #         "FT/T3/U/HL0/Q2/M0", "FT/T3/U/HL0/Q2/M1", "FT/T3/U/HL0/Q2/M2", "FT/T3/U/HL0/Q2/M3", "FT/T3/U/HL0/Q2/M4",
    #         "FT/T3/U/HL1/Q1/M0", "FT/T3/U/HL1/Q1/M1", "FT/T3/U/HL1/Q1/M2", "FT/T3/U/HL1/Q1/M3", "FT/T3/U/HL1/Q1/M4",
    #         "FT/T3/U/HL1/Q3/M0", "FT/T3/U/HL1/Q3/M1", "FT/T3/U/HL1/Q3/M2", "FT/T3/U/HL1/Q3/M3", "FT/T3/U/HL1/Q3/M4"]

def get_unc(input):
    n_files = len(input)
    Tx_chi2 = [[] for _ in range(n_files)]
    Ty_chi2 = [[] for _ in range(n_files)]
    Tz_chi2 = [[] for _ in range(n_files)]

    Rx_chi2 = [[] for _ in range(n_files)]
    Ry_chi2 = [[] for _ in range(n_files)]
    Rz_chi2 = [[] for _ in range(n_files)]
    for i in range(n_files):
        # print('#################################################')
        # print(f'input[{i}]["Tx_chi2"]', input[i]['Tx_chi2'])
        # print(f'input[{i}]["Tx_chi2"][0]', input[i]['Tx_chi2'][0])
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

# for Tx
align_outputs_Tx=[open_alignment(thisfile) for thisfile in files_Tx]
plotted_alignables_Tx=[]
for align_block in align_outputs_Tx:
    thislist_Tx=[]
    for key in align_block.keys():
        thislist_Tx.append(key)
    plotted_alignables_Tx.append(thislist_Tx)
align_outputs_Tx=[convertGlobal(align_block,plotted_alignables_Tx[0]) for align_block in align_outputs_Tx]

# for Ty
align_outputs_Ty=[open_alignment(thisfile) for thisfile in files_Ty]
plotted_alignables_Ty=[]
for align_block in align_outputs_Ty:
    thislist_Ty=[]
    for key in align_block.keys():
        thislist_Ty.append(key)
    plotted_alignables_Ty.append(thislist_Ty)
align_outputs_Ty=[convertGlobal(align_block,plotted_alignables_Ty[0]) for align_block in align_outputs_Ty]

# for Rx
align_outputs_Rx=[open_alignment(thisfile) for thisfile in files_Rx]
plotted_alignables_Rx=[]
for align_block in align_outputs_Rx:
    thislist_Rx=[]
    for key in align_block.keys():
        thislist_Rx.append(key)
    plotted_alignables_Rx.append(thislist_Rx)
align_outputs_Rx=[convertGlobal(align_block,plotted_alignables_Rx[0]) for align_block in align_outputs_Rx]

# for Ry
align_outputs_Ry=[open_alignment(thisfile) for thisfile in files_Ry]
plotted_alignables_Ry=[]
for align_block in align_outputs_Ry:
    thislist_Ry=[]
    for key in align_block.keys():
        thislist_Ry.append(key)
    plotted_alignables_Ry.append(thislist_Ry)
align_outputs_Ry=[convertGlobal(align_block,plotted_alignables_Ry[0]) for align_block in align_outputs_Ry]

# Tx
Tx_chi2_tx = get_unc(align_outputs_Tx)[0]
Ty_chi2_tx = get_unc(align_outputs_Tx)[1]
Tz_chi2_tx = get_unc(align_outputs_Tx)[2]
Rx_chi2_tx = get_unc(align_outputs_Tx)[3]
Ry_chi2_tx = get_unc(align_outputs_Tx)[4]
Rz_chi2_tx = get_unc(align_outputs_Tx)[5]

# Ty
Tx_chi2_ty = get_unc(align_outputs_Ty)[0]
Ty_chi2_ty = get_unc(align_outputs_Ty)[1]
Tz_chi2_ty = get_unc(align_outputs_Ty)[2]
Rx_chi2_ty = get_unc(align_outputs_Ty)[3]
Ry_chi2_ty = get_unc(align_outputs_Ty)[4]
Rz_chi2_ty = get_unc(align_outputs_Ty)[5]

# Rx
Tx_chi2_rx = get_unc(align_outputs_Rx)[0]
Ty_chi2_rx = get_unc(align_outputs_Rx)[1]
Tz_chi2_rx = get_unc(align_outputs_Rx)[2]
Rx_chi2_rx = get_unc(align_outputs_Rx)[3]
Ry_chi2_rx = get_unc(align_outputs_Rx)[4]
Rz_chi2_rx = get_unc(align_outputs_Rx)[5]

# Ry
Tx_chi2_ry = get_unc(align_outputs_Ry)[0]
Ty_chi2_ry = get_unc(align_outputs_Ry)[1]
Tz_chi2_ry = get_unc(align_outputs_Ry)[2]
Rx_chi2_ry = get_unc(align_outputs_Ry)[3]
Ry_chi2_ry = get_unc(align_outputs_Ry)[4]
Rz_chi2_ry = get_unc(align_outputs_Ry)[5]

# print('########## from Tx error changes ###########')

Tz_chi2_vals = [2586.76]

# 7mm and 8mm are missing, last entries are 5mm, 6mm, 7mm, 10mm
Tx_chi2_tz = [15692/768, 8182.69/768, 5278.57/768, 5770.66/768, 12196.7/768, 9622.9/768, 10590.6/768]
Ty_chi2_tz = [1127.64/768, 1122.31/768, 1119.17/768, 1120.25/768, 1125.93/768, 1125.58/768, 1126.06/768]
Tz_chi2_tz = [646.69/768, 287.418/768, 161.672/768, 103.47/768, 71.8544/768, 52.791/768, 25.8676/768]
Rx_chi2_tz = [3086.38/768, 3086.38/768, 3086.38/768, 3086.38/768, 3086.38/768, 3086.38/768, 3086.38/768]
Ry_chi2_tz = [0.00366032/768, 0.0036605/768, 0.00369746/768, 0.00370069/768, 0.0036554/768, 0.00368278/768, 0.0036503/768]
Rz_chi2_tz = [708.737/768, 679.203/768, 658.385/768, 658.059/768, 697.696/768, 678.97/768,  687.408/768]

correct_order = [2, 0, 1, 3, 6, 4, 5, 7, 10, 8, 9, 11]
Tx_err = [0.2, 0.4, 0.6, 0.8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # in [mm], 1 mm is base
Ty_err = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
Tz_err = [2,3,4,5,6,7,10]

Rx_err = [0.2, 0.3, 0.4, 0.5] # mrad
Ry_err = [0.4, 0.3, 0.2, 0.1, 0.02, 0.003, 0.002, 0.001, 0.0008, 0.0006, 0.0005, 0.00044, 0.0004, 0.0002]

total_n_dofs = 768

plt.plot(Tx_err, Tx_chi2_tx / total_n_dofs, marker='x', linestyle='-', label='Tx_chi')
plt.plot(Tx_err, Ty_chi2_tx / total_n_dofs, marker='x', linestyle='', label='Ty_chi')
plt.plot(Tx_err, Tz_chi2_tx / total_n_dofs, marker='x', linestyle='', label='Tz_chi')
plt.plot(Tx_err, Rx_chi2_tx / total_n_dofs, marker='x', linestyle='', label='Rx_chi')
plt.plot(Tx_err, Ry_chi2_tx / total_n_dofs, marker='x', linestyle='', label='Ry_chi')
plt.plot(Tx_err, Rz_chi2_tx / total_n_dofs, marker='x', linestyle='', label='Rz_chi')
plt.hlines(1, 0.2, 10, 'black', 'dotted')
plt.legend()
plt.grid()
plt.title('chi2 after alignment and changed Tx error')
plt.xlabel('[mm]')
# plt.show()
plt.savefig('chi2_plots/Tx_out.pdf')
plt.clf()

plt.plot(Ty_err, Tx_chi2_ty / total_n_dofs, marker='x', linestyle='-', label='Tx_chi')
plt.plot(Ty_err, Ty_chi2_ty / total_n_dofs, marker='x', linestyle='-', label='Ty_chi')
plt.plot(Ty_err, Tz_chi2_ty / total_n_dofs, marker='x', linestyle='', label='Tz_chi')
plt.plot(Ty_err, Rx_chi2_ty / total_n_dofs, marker='x', linestyle='', label='Rx_chi')
plt.plot(Ty_err, Ry_chi2_ty / total_n_dofs, marker='x', linestyle='', label='Ry_chi')
plt.plot(Ty_err, Rz_chi2_ty / total_n_dofs, marker='x', linestyle='', label='Rz_chi')
plt.hlines(1, 1, 10, 'black', 'dotted')
plt.legend()
plt.grid()
plt.title('chi2 after alignment and changed Ty error')
plt.xlabel('[mm]')
# plt.show()
plt.savefig('chi2_plots/Ty_out.pdf')
plt.clf()

plt.plot(Tz_err, Tx_chi2_tz, marker='x', linestyle='-', label='Tx_chi')
plt.plot(Tz_err, Ty_chi2_tz, marker='x', linestyle='', label='Ty_chi')
plt.plot(Tz_err, Tz_chi2_tz, marker='x', linestyle='-', label='Tz_chi')
plt.plot(Tz_err, Rx_chi2_tz, marker='x', linestyle='', label='Rx_chi')
plt.plot(Tz_err, Ry_chi2_tz, marker='x', linestyle='', label='Ry_chi')
plt.plot(Tz_err, Rz_chi2_tz, marker='x', linestyle='', label='Rz_chi')
plt.hlines(1, 2, 10, 'black', 'dotted')
# plt.vlines(x, ymin, ymax, 'red', 'dashed')
plt.legend()
plt.grid()
plt.title('chi2 after alignment and changed Tz error')
plt.xlabel('[mm]')
# plt.show()
plt.savefig('chi2_plots/Tz_out.pdf')
plt.clf()

# Rx
plt.plot(Rx_err, Tx_chi2_rx / total_n_dofs, marker='x', linestyle='-', label='Tx_chi')
plt.plot(Rx_err, Ty_chi2_rx / total_n_dofs, marker='x', linestyle='', label='Ty_chi')
plt.plot(Rx_err, Tz_chi2_rx / total_n_dofs, marker='x', linestyle='', label='Tz_chi')
plt.plot(Rx_err, Rx_chi2_rx / total_n_dofs, marker='x', linestyle='-', label='Rx_chi')
plt.plot(Rx_err, Ry_chi2_rx / total_n_dofs, marker='x', linestyle='', label='Ry_chi')
plt.plot(Rx_err, Rz_chi2_rx / total_n_dofs, marker='x', linestyle='', label='Rz_chi')
plt.hlines(1, 0.2, 0.5, 'black', 'dotted')
plt.legend()
plt.grid()
plt.title('chi2 after alignment and changed Rx error')
plt.xlabel('[mrad]')
# plt.show()
plt.savefig('chi2_plots/Rx_out.pdf')
plt.clf()

# Ry
ry_e = np.linspace(0, 14, 14)
plt.plot(ry_e, Tx_chi2_ry / total_n_dofs, marker='x', linestyle='-', label='Tx_chi')
plt.plot(ry_e, Ty_chi2_ry / total_n_dofs, marker='x', linestyle='', label='Ty_chi')
plt.plot(ry_e, Tz_chi2_ry / total_n_dofs, marker='x', linestyle='', label='Tz_chi')
plt.plot(ry_e, Rx_chi2_ry / total_n_dofs, marker='x', linestyle='', label='Rx_chi')
plt.plot(ry_e, Ry_chi2_ry / total_n_dofs, marker='x', linestyle='-', label='Ry_chi')
plt.plot(ry_e, Rz_chi2_ry / total_n_dofs, marker='x', linestyle='', label='Rz_chi')
plt.hlines(1, -0.2, 14.2, 'black', 'dotted')
plt.legend()
plt.grid()
plt.xticks(ry_e, Ry_err, rotation=45)
plt.title('chi2 after alignment and changed Ry error')
plt.xlabel('[mrad]')
# plt.show()
plt.savefig('chi2_plots/Ry_out.pdf')
plt.clf()

errors_v1 = [0.001, 0.0015, 0.004, 0.0002, 0.0002, 0.0002]
errors_v2 = [0.0005, 0.002, 0.002, 0.0002, 0.0002, 0.0002]
errors_v3 = [0.0002, 0.002, 0.002, 0.0002, 0.0002, 0.0002]
errors_v4 = [0.00018, 0.0015, 0.0018, 0.0004, 0.00000044, 0.00019]
