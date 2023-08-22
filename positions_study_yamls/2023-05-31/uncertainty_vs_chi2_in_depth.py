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

outname_prefix = 'retest_uncertainty/'

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
        # x = x_range[0:range-1] # excldue last value
        # y = (data[index][0:range-1] / dofs).T[0]
        x = x_range
        y = (data[index] / dofs).T[0]
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
        plt.xlabel(r'$[\mu m]$')
    if label in ['Rx', 'Ry', 'Rz']:
        if label == 'Rx':
            index = 3
        if label == 'Ry':
            index = 4
        if label == 'Rz':
            index = 5
        # x = x_range[0:range-1] # excldue last value
        # y = (data[index][0:range-1] / dofs).T[0]
        x = x_range
        y = (data[index] / dofs).T[0]
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
        plt.xlabel('[mrad]')
    plt.hlines(1, min(x_range), max(x_range), 'black')
    # plt.hlines(1, min(x_range), max(x_range), 'black')
    plt.legend()
    plt.grid()
    plt.title(f'chi2 per dof changing only {label} uncertainty')
    plt.ylabel('chi2 / dof')
    # plt.show()
    # plt.savefig(f'chi2_plots/retest/{name}_exclude_last_val.pdf')
    plt.savefig(f'chi2_plots/retest/{name}_full_fit.pdf')
    plt.clf()
    a = popt[0]
    b = popt[1]
    c = popt[2]
    x_intersect = x_intersection(a, b, c) # in different coords
    if label in ['Tx', 'Ty', 'Tz']:
        print(f'for {label}: intersection for chi2 / dof = 1:', x_intersect, 'micron')
    else:
        print(f'for {label}: intersection for chi2 / dof = 1:', x_intersect, 'mrad')

def get_data(files, DoF, align_output):
    num_files = len(files)
    iter_num = 0
    deg = DoF

    runs_T1_U = [[] for _ in range(num_files)]
    runs_T1_V = [[] for _ in range(num_files)]
    runs_T1_X1 = [[] for _ in range(num_files)]
    runs_T1_X2 = [[] for _ in range(num_files)]

    runs_T2_U = [[] for _ in range(num_files)]
    runs_T2_V = [[] for _ in range(num_files)]
    runs_T2_X1 = [[] for _ in range(num_files)]
    runs_T2_X2 = [[] for _ in range(num_files)]

    runs_T3_U = [[] for _ in range(num_files)]
    runs_T3_V = [[] for _ in range(num_files)]
    runs_T3_X1 = [[] for _ in range(num_files)]
    runs_T3_X2 = [[] for _ in range(num_files)]

    T1U_PosRot_yml = [[] for _ in range(num_files)]
    T1U_PosRot = [[] for _ in range(num_files)]
    T1V_PosRot_yml = [[] for _ in range(num_files)]
    T1V_PosRot = [[] for _ in range(num_files)]
    T1X1_PosRot_yml = [[] for _ in range(num_files)]
    T1X1_PosRot = [[] for _ in range(num_files)]
    T1X2_PosRot_yml = [[] for _ in range(num_files)]
    T1X2_PosRot = [[] for _ in range(num_files)]

    T2U_PosRot_yml = [[] for _ in range(num_files)]
    T2U_PosRot = [[] for _ in range(num_files)]
    T2V_PosRot_yml = [[] for _ in range(num_files)]
    T2V_PosRot = [[] for _ in range(num_files)]
    T2X1_PosRot_yml = [[] for _ in range(num_files)]
    T2X1_PosRot = [[] for _ in range(num_files)]
    T2X2_PosRot_yml = [[] for _ in range(num_files)]
    T2X2_PosRot = [[] for _ in range(num_files)]

    T3U_PosRot_yml = [[] for _ in range(num_files)]
    T3U_PosRot = [[] for _ in range(num_files)]
    T3V_PosRot_yml = [[] for _ in range(num_files)]
    T3V_PosRot = [[] for _ in range(num_files)]
    T3X1_PosRot_yml = [[] for _ in range(num_files)]
    T3X1_PosRot = [[] for _ in range(num_files)]
    T3X2_PosRot_yml = [[] for _ in range(num_files)]
    T3X2_PosRot = [[] for _ in range(num_files)]

    runs_T1 = ["FT/T1/U/HL0/Q0/M0", "FT/T1/U/HL0/Q0/M1", "FT/T1/U/HL0/Q0/M2", "FT/T1/U/HL0/Q0/M3", "FT/T1/U/HL0/Q0/M4",
               "FT/T1/U/HL0/Q2/M0", "FT/T1/U/HL0/Q2/M1", "FT/T1/U/HL0/Q2/M2", "FT/T1/U/HL0/Q2/M3", "FT/T1/U/HL0/Q2/M4",
               "FT/T1/U/HL1/Q1/M0", "FT/T1/U/HL1/Q1/M1", "FT/T1/U/HL1/Q1/M2", "FT/T1/U/HL1/Q1/M3", "FT/T1/U/HL1/Q1/M4",
               "FT/T1/U/HL1/Q3/M0", "FT/T1/U/HL1/Q3/M1", "FT/T1/U/HL1/Q3/M2", "FT/T1/U/HL1/Q3/M3", "FT/T1/U/HL1/Q3/M4"]

    runs_T2 = ["FT/T2/U/HL0/Q0/M0", "FT/T2/U/HL0/Q0/M1", "FT/T2/U/HL0/Q0/M2", "FT/T2/U/HL0/Q0/M3", "FT/T2/U/HL0/Q0/M4",
               "FT/T2/U/HL0/Q2/M0", "FT/T2/U/HL0/Q2/M1", "FT/T2/U/HL0/Q2/M2", "FT/T2/U/HL0/Q2/M3", "FT/T2/U/HL0/Q2/M4",
               "FT/T2/U/HL1/Q1/M0", "FT/T2/U/HL1/Q1/M1", "FT/T2/U/HL1/Q1/M2", "FT/T2/U/HL1/Q1/M3", "FT/T2/U/HL1/Q1/M4",
               "FT/T2/U/HL1/Q3/M0", "FT/T2/U/HL1/Q3/M1", "FT/T2/U/HL1/Q3/M2", "FT/T2/U/HL1/Q3/M3", "FT/T2/U/HL1/Q3/M4"]

    runs = ["FT/T3/U/HL0/Q0/M0", "FT/T3/U/HL0/Q0/M1", "FT/T3/U/HL0/Q0/M2", "FT/T3/U/HL0/Q0/M3", "FT/T3/U/HL0/Q0/M4",
            "FT/T3/U/HL0/Q2/M0", "FT/T3/U/HL0/Q2/M1", "FT/T3/U/HL0/Q2/M2", "FT/T3/U/HL0/Q2/M3", "FT/T3/U/HL0/Q2/M4",
            "FT/T3/U/HL1/Q1/M0", "FT/T3/U/HL1/Q1/M1", "FT/T3/U/HL1/Q1/M2", "FT/T3/U/HL1/Q1/M3", "FT/T3/U/HL1/Q1/M4",
            "FT/T3/U/HL1/Q3/M0", "FT/T3/U/HL1/Q3/M1", "FT/T3/U/HL1/Q3/M2", "FT/T3/U/HL1/Q3/M3", "FT/T3/U/HL1/Q3/M4"]

    for file in files:
        x = list(range(len(runs)))
        for j in range(0,len(stations)):
            for k in range(0,len(layers)):
                if j==0 and k==0:
                    runs_T1_U[iter_num]=runs_T1
                    runs_T2_U[iter_num]=runs_T2
                    runs_T3_U[iter_num]=runs
                elif j==0 and k==1:
                    for i in range(0,len(runs)):
                        string1 = runs_T1[i]
                        string2 = runs_T2[i]
                        string3 = runs[i]
                        runs_T1_V[iter_num].append(string1.replace("T1/U", "T1/V"))
                        runs_T2_V[iter_num].append(string2.replace("T2/U", "T2/V"))
                        runs_T3_V[iter_num].append(string3.replace("T3/U", "T3/V"))
                elif j==0 and k==2:
                    for i in range(0,len(runs)):
                        string1 = runs_T1[i]
                        string2 = runs_T2[i]
                        string3 = runs[i]
                        runs_T1_X1[iter_num].append(string1.replace("T1/U", "T1/X1"))
                        runs_T2_X1[iter_num].append(string2.replace("T2/U", "T2/X1"))
                        runs_T3_X1[iter_num].append(string3.replace("T3/U", "T3/X1"))
                elif j==0 and k==3:
                    for i in range(0,len(runs)):
                        string1 = runs_T1[i]
                        string2 = runs_T2[i]
                        string3 = runs[i]
                        runs_T1_X2[iter_num].append(string1.replace("T1/U", "T1/X2"))
                        runs_T2_X2[iter_num].append(string2.replace("T2/U", "T2/X2"))
                        runs_T3_X2[iter_num].append(string3.replace("T3/U", "T3/X2"))

        for i in range(0,len(runs)):
            # what is the json file used for
            with open(file, 'r') as stream:  # why do i do this???
                data_loaded = align_output[iter_num]
                # print(data_loaded[runs_T1_U[iter_num][i]].keys())
                # print(deg, data_loaded[runs_T1_U[iter_num][i]].values())

                T1U_PosRot_yml[iter_num].append(data_loaded[runs_T1_U[iter_num][i]][deg])
                T1U_PosRot[iter_num].append(T1U_PosRot_yml[iter_num][i][0])

                T1V_PosRot_yml[iter_num].append(data_loaded[runs_T1_V[iter_num][i]][deg])
                T1V_PosRot[iter_num].append(T1V_PosRot_yml[iter_num][i][0])

                T1X1_PosRot_yml[iter_num].append(data_loaded[runs_T1_X1[iter_num][i]][deg])
                T1X1_PosRot[iter_num].append(T1X1_PosRot_yml[iter_num][i][0])

                T1X2_PosRot_yml[iter_num].append(data_loaded[runs_T1_X2[iter_num][i]][deg])
                T1X2_PosRot[iter_num].append(T1X2_PosRot_yml[iter_num][i][0])

                # T2
                T2U_PosRot_yml[iter_num].append(data_loaded[runs_T2_U[iter_num][i]][deg])
                T2U_PosRot[iter_num].append(T2U_PosRot_yml[iter_num][i][0])

                T2V_PosRot_yml[iter_num].append(data_loaded[runs_T2_V[iter_num][i]][deg])
                T2V_PosRot[iter_num].append(T2V_PosRot_yml[iter_num][i][0])

                T2X1_PosRot_yml[iter_num].append(data_loaded[runs_T2_X1[iter_num][i]][deg])
                T2X1_PosRot[iter_num].append(T2X1_PosRot_yml[iter_num][i][0])

                T2X2_PosRot_yml[iter_num].append(data_loaded[runs_T2_X2[iter_num][i]][deg])
                T2X2_PosRot[iter_num].append(T2X2_PosRot_yml[iter_num][i][0])

                # T3
                T3U_PosRot_yml[iter_num].append(data_loaded[runs_T3_U[iter_num][i]][deg])
                T3U_PosRot[iter_num].append(T3U_PosRot_yml[iter_num][i][0])

                T3V_PosRot_yml[iter_num].append(data_loaded[runs_T3_V[iter_num][i]][deg])
                T3V_PosRot[iter_num].append(T3V_PosRot_yml[iter_num][i][0])

                T3X1_PosRot_yml[iter_num].append(data_loaded[runs_T3_X1[iter_num][i]][deg])
                T3X1_PosRot[iter_num].append(T3X1_PosRot_yml[iter_num][i][0])

                T3X2_PosRot_yml[iter_num].append(data_loaded[runs_T3_X2[iter_num][i]][deg])
                T3X2_PosRot[iter_num].append(T3X2_PosRot_yml[iter_num][i][0])
        iter_num += 1
    return np.array(T1U_PosRot), np.array(T1V_PosRot), np.array(T1X1_PosRot), np.array(T1X2_PosRot), np.array(T2U_PosRot), np.array(T2V_PosRot), np.array(T2X1_PosRot), np.array(T2X2_PosRot), np.array(T3U_PosRot), np.array(T3V_PosRot), np.array(T3X1_PosRot), np.array(T3X2_PosRot)

def plot(data_arr, survey_pos, outname, run_labels, title_label, layerID):
    max_Q0, max_Q1, max_Q2, max_Q3 = [], [], [], [] # should store 2 values: value and where per layer
    # print('len data:', len(data_arr))
    # change this for own needs as well
    outfiles = 'constants_check_up/'
    total_layer_num = 12
    total_num_runs = len(run_labels) # number of runs
    # print(total_num_runs)
    x = np.linspace(0, 4, 5) # 5 modules per quarter
    '''
        instead of plotting the difference between survey and alignment runs
        also plot diff:
        abs(run 1 - run 2), abs(run 2 - run 3), abs(run 3 - run 4), etc
        run not beeing the LHCb run but the alignment runs
    '''
    L = ['Q2', 'Q3', 'Q0', 'Q1']
    # print(layerID, total_num_runs)
    # print(data_arr)
    # for i in range(total_num_runs):  # when using 'constants'
    for i in range(total_num_runs):  # when using 'compare'
        if survey_pos == 'constants':
            # print('range index i =', i)
            x1 = data_arr[i][0:5]    # Q0
            # print(x1)
            x2 = data_arr[i][5:10]   # Q2
            x3 = data_arr[i][10:15]  # Q1
            x4 = data_arr[i][15:20]  # Q3
        if survey_pos == 'compare':
            x1 = data_arr[i][0:5] - data_arr[i+1][0:5]      # Q0
            x2 = data_arr[i][5:10] - data_arr[i+1][5:10]    # Q2
            x3 = data_arr[i][10:15] - data_arr[i+1][10:15]  # Q1
            x4 = data_arr[i][15:20] - data_arr[i+1][15:20]  # Q3
        if survey_pos == 'survey':
            x1 = data_arr[i][0:5] - survey_pos[i][0:5]     # Q0
            x2 = data_arr[i][5:10] - survey_pos[i][5:10]   # Q2
            x3 = data_arr[i][10:15] - survey_pos[i][10:15] # Q1
            x4 = data_arr[i][15:20] - survey_pos[i][15:20] # Q3

        ax = [plt.subplot(2,2,i+1) for i in range(4)]
        plt.figure()
        count = 0
        for a in ax:
            a.text(0.1, 0.7, L[count], transform=a.transAxes, weight="bold")
            plt.sca(a)
            if count == 0: # Q2
                plt.scatter(x, x2[::-1], color=colors[i], marker=markers[i], s=10)
                plt.ylabel(f'{title_label} [mm]')
                plt.title(f'local {title_label}')
                plt.hlines(0, 0, 5, colors='black', linestyles='dashed')
                a.invert_yaxis()
            if count == 1: # Q3
                plt.scatter(x, x4, color=colors[i], marker=markers[i], s=10, label = f'{run_labels[i]}')
                plt.title(f'layer {layerID}')
                plt.hlines(0, 0, 5, colors='black', linestyles='dashed')
                a.yaxis.tick_right()
                plt.legend(loc='best', fontsize='8')
            if count == 2: # Q0
                plt.scatter(x, x1[::-1], color=colors[i], marker=markers[i], s=10)
                plt.xticks(x, ["Q0M4", "Q0M3", "Q0M2", "Q0M1", "Q0M0"], rotation=45, fontsize=5)
                plt.hlines(0, 0, 5, colors='black', linestyles='dashed')
            if count == 3: # Q1
                plt.scatter(x, x3, color=colors[i], marker=markers[i], s=10)
                a.invert_yaxis()
                plt.hlines(0, 0, 5, colors='black', linestyles='dashed')
                a.yaxis.tick_right()
                plt.xticks(x, ["Q0M0", "Q0M1", "Q0M2", "Q0M3", "Q0M4"], rotation=45, fontsize=5)
            count += 1
        plt.subplots_adjust(wspace=0, hspace=0)
        # l = ['base','Tx','Ty','Rx','Tz','Ry']
        plt.savefig(f'{outname_prefix}{outfiles}' + run_labels[i] + outname + '_' + layerID + '_' + title_label + '.pdf')
        # plt.savefig(f'{outname_prefix}{outfiles}{run_labels[i]}/' + outname + '_' + layerID + '_' + title_label + '.pdf')

    plt.clf()

layers = ['T1U', 'T1V', 'T1X1', 'T1X2', 'T2U', 'T2V', 'T2X1', 'T2X2', 'T3U', 'T3V', 'T3X1', 'T3X2']

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
# chi2_values_from_Rx_changes_OT = get_chi2_values(files_Rx_onlyTx)
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
Ry_err = [0.0003, 0.00035, 0.0004, 0.00044, 0.0005, 0.0006] # micro rad
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
plotting(Rz_err, chi2_values_from_Rz_changes, total_n_dofs, 'Rz', 'only_Rz_variable')

best_Tx = 0.3 # micron,  0.22 fit not working so well
best_Ty = 1.2 # micron
best_Tz = 1.9 # micron, # 1.83
best_Rx = 0.0004 # 0.4 mrad
best_Ry = 0.00000044 # 0.44 micro rad
best_Rz = 0.0002 # mrad

'''
    plot the constants (positions and rotations) for the different
    stages for the tuning
'''

path = 'retest_uncertainty/constants_check_up/input_files'
input_constants = [\
    f'{path}/parsedlog_base.json',
    f'{path}/parsedlog_Tx_0_3_micron.json',
    f'{path}/parsedlog_Ty_1_2_micron.json',
    f'{path}/parsedlog_Rx_0_4_mrad.json',
    f'{path}/parsedlog_Tz_1_9_micron.json',
    f'{path}/parsedlog_Ry_0_44_micro_rad.json',
]

labels_constants = [\
    'base',
    'Tx',
    'TxTy',
    'TxTyRx',
    'TxTyTzRx',
    'all',
]
constant_diff = [\
    'base-Tx',
    'Tx-Ty',
    'Ty-Rx',
    'Rx-Tz',
    'Tz-all'
]

align_outputs=[open_alignment(thisfile) for thisfile in input_constants]
plotted_alignables=[]
for align_block in align_outputs:
    thislist=[]
    for key in align_block.keys():
        if "FT" in key:
            thislist.append(key)
    plotted_alignables.append(thislist)
align_outputs=[convertGlobal(align_block,plotted_alignables[0]) for align_block in align_outputs]

tx = get_data(input_constants, 'Tx', align_outputs)
ty = get_data(input_constants, 'Ty', align_outputs)
tz = get_data(input_constants, 'Tz', align_outputs)
x_glob = get_data(input_constants, 'x_global', align_outputs)
y_glob = get_data(input_constants, 'y_global', align_outputs)
z_glob = get_data(input_constants, 'z_global', align_outputs)
nHits = get_data(input_constants, 'nHits', align_outputs)
nTracks = get_data(input_constants, 'nTracks', align_outputs)

for n in range(12):
    tx_data = tx[n]
    ty_data = ty[n]
    tz_data = tz[n]
    x_g = x_glob[n]
    y_g = y_glob[n]
    z_g = z_glob[n]
    # print(tx_data)
    # plot(tx_data, 'constants', 'plain_constants', labels_constants, 'Tx', layers[n])  # set [] to survey Tx if i want to compare to survey positions
    plot(tx_data, 'compare', 'diff_tuned_params', constant_diff, 'Tx', layers[n])
