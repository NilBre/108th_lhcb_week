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

# change to your own output directories
outname_prefix = 'SciFiAlignv3/'

def hits_vs_tracks(arr1, arr2, labels, layerID):
    keys = list(align_outputs[0]['FT/T1UHL0/Q0M0'].keys())
    outfiles = 'hits_and_tracks/'
    # arr has 8 arrays, 1 for each datafile
    for i in range(len(arr1)):
        plt.plot(arr1, arr2, label = f'{labels[i]}')
        if plt.grid(True):
            plt.grid()
        plt.legend(loc='best')
        plt.xlabel(f'{keys[10]}')
        plt.ylabel(f'{keys[9]}')
        plt.title('nHits vs nTracks')
    plt.savefig(f'{outname_prefix}{outfiles}' + 'nHitsVsnTracks_' + layerID + '.pdf')
    plt.clf()

def plot_with_globals(data_arr, outname, run_labels, layer_names, glob_data1, glob_data2):
    # this as well

    outfiles = 'outfiles_vs_global/'
    total_layer_num = len(layer_names)
    total_num_runs = len(run_labels)

    x_data = data_arr
    x_glob = glob_data2
    z_glob = glob_data1

    z_positions = [] # 12 values, 1 for each layer
    for j in range(total_layer_num):
        z_positions.append(z_glob[j][0][0])

    x_shifted = np.array(x_glob) + np.array(x_data)
    x_means = [[] for _ in range(total_num_runs)]

    for run in range(total_num_runs):
        for layer in range(total_layer_num):
            x_means[run].append(np.mean(x_shifted[layer][run]))

    # change from json  order (U, V, X1, X2) to physical (X1, U, V, X2)
    correct_order = [2, 0, 1, 3, 6, 4, 5, 7, 10, 8, 9, 11]

    for runs in range(total_num_runs):
        correct_x_order = [x_means[runs][iter] for iter in correct_order]
        plt.errorbar(z_positions, correct_x_order, yerr=sem(x_means[runs]), ls='', marker='x', c=colors[runs], label=f'{run_labels[runs]}')
        if plt.grid(True):
            plt.grid()
        plt.legend(loc='best')
        plt.ylabel('mean Tx module position')
        plt.title('mean Tx vs. global z, after V3 alignment, magDown')
        plt.xticks(z_positions, ['T1U', 'T1V', 'T1X1', 'T1X2', 'T2U', 'T2V', 'T2X1', 'T2X2', 'T3U', 'T3V', 'T3X1', 'T3X2'], rotation=45, fontsize=10)
    plt.savefig(f'{outname_prefix}/{outfiles}' + 'all_runs_' + outname + '.pdf')
    plt.clf()

def compare_alignments(comparison_data, outname, run_labels, title_label, layerID):
    outfiles = 'outfiles_comparison/'
    base = comparison_data[0]
    diff = [[] for _ in range(len(run_labels) - 1)]
    for n in range(len(comparison_data) - 1):
        diff[n].append(np.array(base) - np.array(comparison_data[n+1]))

    x = np.linspace(0, 5, 5)

    L = ['Q2', 'Q3', 'Q0', 'Q1']
    for i in range(len(comparison_data) - 1):
        # x1 and x2 need to be reversed since Q0 is on the inside!
        x1 = diff[i][0][0:5] # Q0
        x2 = diff[i][0][5:10] # Q2
        x3 = diff[i][0][10:15] # Q1
        x4 = diff[i][0][15:20] # Q3

        ax = [plt.subplot(2,2,i+1) for i in range(4)]
        plt.figure()
        count = 0
        for a in ax:
            a.text(0.1, 0.7, L[count], transform=a.transAxes, weight="bold")
            plt.sca(a)
            if count == 0: # Q2
                plt.scatter(x, x2[::-1], color=colors[i], marker=markers[i], s=10)
                plt.ylabel(f'{title_label} [mm]')
                plt.hlines(0, 0, 5, colors='black', linestyles='dashed')
                plt.title(f'module difference compared to run 256145')
                # a.invert_yaxis()
            if count == 1: # Q3
                plt.scatter(x, x4, color=colors[i], marker=markers[i], s=10, label = f'{run_labels[i+1]}')
                plt.title(f'layer {layerID}')
                plt.hlines(0, 0, 5, colors='black', linestyles='dashed')
                a.yaxis.tick_right()
                plt.legend(loc='best')
            if count == 2: # Q0
                plt.scatter(x, x1[::-1], color=colors[i], marker=markers[i], s=10)
                plt.xticks(x, ["T3UHL0Q0M4", "T3UHL0Q0M3", "T3UHL0Q0M2", "T3UHL0Q0M1", "T3UHL0Q0M0"], rotation=45, fontsize=5)
                plt.hlines(0, 0, 5, colors='black', linestyles='dashed')
                # a.invert_yaxis()
            if count == 3: # Q1
                plt.scatter(x, x3, color=colors[i], marker=markers[i], s=10)
                a.yaxis.tick_right()
                plt.hlines(0, 0, 5, colors='black', linestyles='dashed')
                # a.invert_yaxis()
                plt.xticks(x, ["T3UHL0Q0M0", "T3UHL0Q0M1", "T3UHL0Q0M2", "T3UHL0Q0M3", "T3UHL0Q0M4"], rotation=45, fontsize=5)
            count += 1
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.savefig(f'{outname_prefix}{outfiles}' + outname + '_diff_plots_' + layerID + '.pdf')
    plt.clf()


def plot(data_arr, survey_pos, outname, run_labels, title_label, layerID):
    max_Q0, max_Q1, max_Q2, max_Q3 = [], [], [], [] # should store 2 values: value and where per layer
    print('len data:', len(data_arr))
    # change this for own needs as well
    outfiles = 'relative_pos/'
    total_layer_num = 12 # number of layers
    total_num_runs = len(run_labels)
    # print(total_num_runs)
    x = np.linspace(0, 5, 5)
    '''
        instead of plotting the difference between survey and alignment runs
        also plot diff:
        abs(run 1 - run 2), abs(run 2 - run 3), abs(run 3 - run 4), etc
        run not beeing the LHCb run but the alignment runs
    '''
    L = ['Q2', 'Q3', 'Q0', 'Q1']
    for i in range(total_num_runs-1):
        if len(survey_pos) == 0:
            x1 = data_arr[i][0:5] - data_arr[i+1][0:5]  # Q0
            x2 = data_arr[i][5:10] - data_arr[i+1][5:10]  # Q2
            x3 = data_arr[i][10:15] - data_arr[i+1][10:15]  # Q1
            x4 = data_arr[i][15:20] - data_arr[i+1][15:20]  # Q3
        # if len(survey_pos) == 0:
        #     x1 = data_arr[i][0:5]  # Q0
        #     x2 = data_arr[i][5:10]  # Q2
        #     x3 = data_arr[i][10:15]  # Q1
        #     x4 = data_arr[i][15:20]  # Q3
        else:
            x1 = data_arr[i][0:5] - survey_pos[i][0:5] # Q0
            x2 = data_arr[i][5:10] - survey_pos[i][5:10] # Q2
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
        plt.savefig(f'{outname_prefix}{outfiles}' + outname + '_' + layerID + '_' + title_label + '.pdf')

    plt.clf()

def calculateDiff(align_output1,align_output2,plotted_alignables,absolute=False):
    diff_result={}
    for alignable in plotted_alignables:
        diff_result[alignable]={}
        for label in align_output1[alignable].keys():
            if len(label)>2:
                if "global" in label:
                    diff_result[alignable][label]=align_output1[alignable][label]
                    continue
                else:
                    continue
            diff_result[alignable][label]=[]
            for iternum in range(0,len(align_output1[alignable][label])):
                if absolute:
                    diff_result[alignable][label].append(abs(align_output2[alignable][label][iternum]-align_output1[alignable][label][iternum]))
                else:
                    diff_result[alignable][label].append(align_output2[alignable][label][iternum]-align_output1[alignable][label][iternum])

    return diff_result

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

def makeModulesAlignLogFormat(filename_in,thistype="output",maxindex=0):
    modules=[]
    linenno=0
    index=0

    align_output={}
    with open(filename_in,"r") as inputfile:
        mylabel=""
        for line in inputfile:
            index=index+1
            if index<maxindex:
                continue
            if regex_typelabel.search(line):
                quarterlabel=line.split('"')[3]
                layer=quarterlabel[2:quarterlabel.index("Q")]
                quarternum=int(quarterlabel[-3:-2])
            if regex_amodule.search(line):
                if thistype=="output":
                    text=line.split(">")
                    valueX=text[2].split()[0]
                    valueY=text[2].split()[1]
                    valueZ=text[2].split()[2]
                    valueZ=valueZ.split("<")[0]
                    align_output[quarterlabel]={}
                    align_output[quarterlabel]["Tx"]=[float(valueX)]
                    align_output[quarterlabel]["Ty"]=[float(valueY)]
                    align_output[quarterlabel]["Tz"]=[float(valueZ)]
                elif thistype=="input":
                    text=line.split(">")
                    valueX=text[1].split()[0]
                    valueY=text[1].split()[1]
                    valueZ=text[1].split()[2]
                    valueZ=valueZ.split("<")[0]
                    align_output[quarterlabel]={}
                    align_output[quarterlabel]["Tx"]=[float(valueX)]
                    align_output[quarterlabel]["Ty"]=[float(valueY)]
                    align_output[quarterlabel]["Tz"]=[float(valueZ)]
            if regex_rmodule.search(line):
                if thistype=="output":
                    text=line.split(">")
                    valueX=text[4].split()[0]
                    valueY=text[4].split()[1]
                    valueZ=text[4].split()[2]
                    valueZ=valueZ.split("<")[0]
                    align_output[quarterlabel]["Rx"]=[1000*float(valueX)]
                    align_output[quarterlabel]["Ry"]=[1000*float(valueY)]
                    align_output[quarterlabel]["Rz"]=[1000*float(valueZ)]
                elif thistype=="input":
                    text=line.split(">")
                    valueX=text[1].split()[0]
                    valueY=text[1].split()[1]
                    valueZ=text[1].split()[2]
                    valueZ=valueZ.split("<")[0]
                    align_output[quarterlabel]["Rx"]=[1000*float(valueX)]
                    align_output[quarterlabel]["Ry"]=[1000*float(valueY)]
                    align_output[quarterlabel]["Rz"]=[1000*float(valueZ)]

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

    runs_T1 = ["FT/T1UHL0/Q0M0", "FT/T1UHL0/Q0M1", "FT/T1UHL0/Q0M2", "FT/T1UHL0/Q0M3", "FT/T1UHL0/Q0M4",
            "FT/T1UHL0/Q2M0", "FT/T1UHL0/Q2M1", "FT/T1UHL0/Q2M2", "FT/T1UHL0/Q2M3", "FT/T1UHL0/Q2M4",
            "FT/T1UHL1/Q1M0", "FT/T1UHL1/Q1M1", "FT/T1UHL1/Q1M2", "FT/T1UHL1/Q1M3", "FT/T1UHL1/Q1M4",
            "FT/T1UHL1/Q3M0", "FT/T1UHL1/Q3M1", "FT/T1UHL1/Q3M2", "FT/T1UHL1/Q3M3", "FT/T1UHL1/Q3M4"]

    runs_T2 = ["FT/T2UHL0/Q0M0", "FT/T2UHL0/Q0M1", "FT/T2UHL0/Q0M2", "FT/T2UHL0/Q0M3", "FT/T2UHL0/Q0M4",
               "FT/T2UHL0/Q2M0", "FT/T2UHL0/Q2M1", "FT/T2UHL0/Q2M2", "FT/T2UHL0/Q2M3", "FT/T2UHL0/Q2M4",
               "FT/T2UHL1/Q1M0", "FT/T2UHL1/Q1M1", "FT/T2UHL1/Q1M2", "FT/T2UHL1/Q1M3", "FT/T2UHL1/Q1M4",
               "FT/T2UHL1/Q3M0", "FT/T2UHL1/Q3M1", "FT/T2UHL1/Q3M2", "FT/T2UHL1/Q3M3", "FT/T2UHL1/Q3M4"]

    runs = ["FT/T3UHL0/Q0M0", "FT/T3UHL0/Q0M1", "FT/T3UHL0/Q0M2", "FT/T3UHL0/Q0M3", "FT/T3UHL0/Q0M4",
            "FT/T3UHL0/Q2M0", "FT/T3UHL0/Q2M1", "FT/T3UHL0/Q2M2", "FT/T3UHL0/Q2M3", "FT/T3UHL0/Q2M4",
            "FT/T3UHL1/Q1M0", "FT/T3UHL1/Q1M1", "FT/T3UHL1/Q1M2", "FT/T3UHL1/Q1M3", "FT/T3UHL1/Q1M4",
            "FT/T3UHL1/Q3M0", "FT/T3UHL1/Q3M1", "FT/T3UHL1/Q3M2", "FT/T3UHL1/Q3M3", "FT/T3UHL1/Q3M4"]

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
                        runs_T1_V[iter_num].append(string1.replace("T1U", "T1V"))
                        runs_T2_V[iter_num].append(string2.replace("T2U", "T2V"))
                        runs_T3_V[iter_num].append(string3.replace("T3U", "T3V"))
                elif j==0 and k==2:
                    for i in range(0,len(runs)):
                        string1 = runs_T1[i]
                        string2 = runs_T2[i]
                        string3 = runs[i]
                        runs_T1_X1[iter_num].append(string1.replace("T1U", "T1X1"))
                        runs_T2_X1[iter_num].append(string2.replace("T2U", "T2X1"))
                        runs_T3_X1[iter_num].append(string3.replace("T3U", "T3X1"))
                elif j==0 and k==3:
                    for i in range(0,len(runs)):
                        string1 = runs_T1[i]
                        string2 = runs_T2[i]
                        string3 = runs[i]
                        runs_T1_X2[iter_num].append(string1.replace("T1U", "T1X2"))
                        runs_T2_X2[iter_num].append(string2.replace("T2U", "T2X2"))
                        runs_T3_X2[iter_num].append(string3.replace("T3U", "T3X2"))

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

def get_survey_data(file, dof, spatial_object):
    iter_num = 0
    dof_value = 0
    PosRot = spatial_object # position or rotation
    if dof == 'Tx' or dof == 'Rx':
        dof_value = 0
    if dof == 'Ty' or dof == 'Ry':
        dof_value = 1
    if dof == 'Tz' or dof == 'Rz':
        dof_value = 2

    runs_T1_U = []
    runs_T1_V = []
    runs_T1_X1 = []
    runs_T1_X2 = []

    runs_T2_U = []
    runs_T2_V = []
    runs_T2_X1 = []
    runs_T2_X2 = []

    runs_T3_U = []
    runs_T3_V = []
    runs_T3_X1 = []
    runs_T3_X2 = []

    T1U_PosRot_yml = []
    T1U_PosRot = []
    T1V_PosRot_yml = []
    T1V_PosRot = []
    T1X1_PosRot_yml = []
    T1X1_PosRot = []
    T1X2_PosRot_yml = []
    T1X2_PosRot = []

    T2U_PosRot_yml = []
    T2U_PosRot = []
    T2V_PosRot_yml = []
    T2V_PosRot = []
    T2X1_PosRot_yml = []
    T2X1_PosRot = []
    T2X2_PosRot_yml = []
    T2X2_PosRot = []

    T3U_PosRot_yml = []
    T3U_PosRot = []
    T3V_PosRot_yml = []
    T3V_PosRot = []
    T3X1_PosRot_yml = []
    T3X1_PosRot = []
    T3X2_PosRot_yml = []
    T3X2_PosRot = []

    runs_T1 = ["T1UHL0Q0M0", "T1UHL0Q0M1", "T1UHL0Q0M2", "T1UHL0Q0M3", "T1UHL0Q0M4",
               "T1UHL0Q2M0", "T1UHL0Q2M1", "T1UHL0Q2M2", "T1UHL0Q2M3", "T1UHL0Q2M4",
               "T1UHL1Q1M0", "T1UHL1Q1M1", "T1UHL1Q1M2", "T1UHL1Q1M3", "T1UHL1Q1M4",
               "T1UHL1Q3M0", "T1UHL1Q3M1", "T1UHL1Q3M2", "T1UHL1Q3M3", "T1UHL1Q3M4"]

    runs_T2 = ["T2UHL0Q0M0", "T2UHL0Q0M1", "T2UHL0Q0M2", "T2UHL0Q0M3", "T2UHL0Q0M4",
               "T2UHL0Q2M0", "T2UHL0Q2M1", "T2UHL0Q2M2", "T2UHL0Q2M3", "T2UHL0Q2M4",
               "T2UHL1Q1M0", "T2UHL1Q1M1", "T2UHL1Q1M2", "T2UHL1Q1M3", "T2UHL1Q1M4",
               "T2UHL1Q3M0", "T2UHL1Q3M1", "T2UHL1Q3M2", "T2UHL1Q3M3", "T2UHL1Q3M4"]

    runs = ["T3UHL0Q0M0", "T3UHL0Q0M1", "T3UHL0Q0M2", "T3UHL0Q0M3", "T3UHL0Q0M4",
            "T3UHL0Q2M0", "T3UHL0Q2M1", "T3UHL0Q2M2", "T3UHL0Q2M3", "T3UHL0Q2M4",
            "T3UHL1Q1M0", "T3UHL1Q1M1", "T3UHL1Q1M2", "T3UHL1Q1M3", "T3UHL1Q1M4",
            "T3UHL1Q3M0", "T3UHL1Q3M1", "T3UHL1Q3M2", "T3UHL1Q3M3", "T3UHL1Q3M4"]

    x = list(range(len(runs)))
    for j in range(0,len(stations)):
        for k in range(0,len(layers)):
            if j==0 and k==0:
                runs_T1_U=runs_T1
                runs_T2_U=runs_T2
                runs_T3_U=runs
            elif j==0 and k==1:
                for i in range(0,len(runs)):
                    string1 = runs_T1[i]
                    string2 = runs_T2[i]
                    string3 = runs[i]
                    runs_T1_V.append(string1.replace("T1U", "T1V"))
                    runs_T2_V.append(string2.replace("T2U", "T2V"))
                    runs_T3_V.append(string3.replace("T3U", "T3V"))
            elif j==0 and k==2:
                for i in range(0,len(runs)):
                    string1 = runs_T1[i]
                    string2 = runs_T2[i]
                    string3 = runs[i]
                    runs_T1_X1.append(string1.replace("T1U", "T1X1"))
                    runs_T2_X1.append(string2.replace("T2U", "T2X1"))
                    runs_T3_X1.append(string3.replace("T3U", "T3X1"))
            elif j==0 and k==3:
                for i in range(0,len(runs)):
                    string1 = runs_T1[i]
                    string2 = runs_T2[i]
                    string3 = runs[i]
                    runs_T1_X2.append(string1.replace("T1U", "T1X2"))
                    runs_T2_X2.append(string2.replace("T2U", "T2X2"))
                    runs_T3_X2.append(string3.replace("T3U", "T3X2"))
    # print(runs_T1_U, runs_T1_X2)
    for i in range(0,len(runs)):
        with open(file, 'r') as stream:
            data_loaded = yaml.load(stream, Loader=yaml.Loader)
            T1U_PosRot_yml.append(data_loaded[runs_T1_U[i]][PosRot][dof_value])
            T1U_PosRot.append(float(re.findall(r'\d+',T1U_PosRot_yml[i])[0] + "." + re.findall(r'\d+',T1U_PosRot_yml[i])[1]))

            T1V_PosRot_yml.append(data_loaded[runs_T1_V[i]][PosRot][dof_value])
            T1V_PosRot.append(float(re.findall(r'\d+',T1V_PosRot_yml[i])[0] + "." + re.findall(r'\d+',T1V_PosRot_yml[i])[1]))

            T1X1_PosRot_yml.append(data_loaded[runs_T1_X1[i]][PosRot][dof_value])
            T1X1_PosRot.append(float(re.findall(r'\d+',T1X1_PosRot_yml[i])[0] + "." + re.findall(r'\d+',T1X1_PosRot_yml[i])[1]))

            T1X2_PosRot_yml.append(data_loaded[runs_T1_X2[i]][PosRot][dof_value])
            T1X2_PosRot.append(float(re.findall(r'\d+',T1X2_PosRot_yml[i])[0] + "." + re.findall(r'\d+',T1X2_PosRot_yml[i])[1]))

            # T2
            T2U_PosRot_yml.append(data_loaded[runs_T2_U[i]][PosRot][dof_value])
            T2U_PosRot.append(float(re.findall(r'\d+',T2U_PosRot_yml[i])[0] + "." + re.findall(r'\d+',T2U_PosRot_yml[i])[1]))

            T2V_PosRot_yml.append(data_loaded[runs_T2_V[i]][PosRot][dof_value])
            T2V_PosRot.append(float(re.findall(r'\d+',T2V_PosRot_yml[i])[0] + "." + re.findall(r'\d+',T2V_PosRot_yml[i])[1]))

            T2X1_PosRot_yml.append(data_loaded[runs_T2_X1[i]][PosRot][dof_value])
            T2X1_PosRot.append(float(re.findall(r'\d+',T2X1_PosRot_yml[i])[0] + "." + re.findall(r'\d+',T2X1_PosRot_yml[i])[1]))

            T2X2_PosRot_yml.append(data_loaded[runs_T2_X2[i]][PosRot][dof_value])
            T2X2_PosRot.append(float(re.findall(r'\d+',T2X2_PosRot_yml[i])[0] + "." + re.findall(r'\d+',T2X2_PosRot_yml[i])[1]))

            # T3
            T3U_PosRot_yml.append(data_loaded[runs_T3_U[i]][PosRot][dof_value])
            T3U_PosRot.append(float(re.findall(r'\d+',T3U_PosRot_yml[i])[0] + "." + re.findall(r'\d+',T3U_PosRot_yml[i])[1]))

            T3V_PosRot_yml.append(data_loaded[runs_T3_V[i]][PosRot][dof_value])
            T3V_PosRot.append(float(re.findall(r'\d+',T3V_PosRot_yml[i])[0] + "." + re.findall(r'\d+',T3V_PosRot_yml[i])[1]))

            T3X1_PosRot_yml.append(data_loaded[runs_T3_X1[i]][PosRot][dof_value])
            T3X1_PosRot.append(float(re.findall(r'\d+',T3X1_PosRot_yml[i])[0] + "." + re.findall(r'\d+',T3X1_PosRot_yml[i])[1]))

            T3X2_PosRot_yml.append(data_loaded[runs_T3_X2[i]][PosRot][dof_value])
            T3X2_PosRot.append(float(re.findall(r'\d+',T3X2_PosRot_yml[i])[0] + "." + re.findall(r'\d+',T3X2_PosRot_yml[i])[1]))

        # data is filled into the correct lists
    return np.array(T1U_PosRot), np.array(T1V_PosRot), np.array(T1X1_PosRot), np.array(T1X2_PosRot), np.array(T2U_PosRot), np.array(T2V_PosRot), np.array(T2X1_PosRot), np.array(T2X2_PosRot), np.array(T3U_PosRot), np.array(T3V_PosRot), np.array(T3X1_PosRot), np.array(T3X2_PosRot)

def max_module_deviation(data_arr, identifier, run_labels, layerID):
    max_deviation = [{} for _ in range(total_num_runs-1)] # for 12 layers
    max_Q0, max_Q1, max_Q2, max_Q3 = [], [], [], [] # should store 2 values: value and where per layer
    # change this for own needs as well
    outfiles = 'relative_pos/'
    total_layer_num = 12 # number of layers
    total_num_runs = len(run_labels)

    L = ['Q2', 'Q3', 'Q0', 'Q1']
    for i in range(total_num_runs-1):
        x1 = data_arr[i][0:5] - data_arr[i+1][0:5]  # Q0
        x2 = data_arr[i][5:10] - data_arr[i+1][5:10]  # Q2
        x3 = data_arr[i][10:15] - data_arr[i+1][10:15]  # Q1
        x4 = data_arr[i][15:20] - data_arr[i+1][15:20]  # Q3
        max_deviation[i][f'diff_{i}'] = {
        'Q0': max(x1), 'Module': np.argmax(x1),
        'Q1': max(x3), 'Module': np.argmax(x3),
        'Q2': max(x2), 'Module': np.argmax(x2),
        'Q3': max(x4), 'Module': np.argmax(x4)
        }
    return max_deviation
# hep.style.use(hep.style.LHCb2)

# import matplotlib.patches as mpl_patches

def meta_constructor(loader, node):
   return loader.construct_mapping(node)

yaml.add_constructor('!alignment', meta_constructor)

files = [\
         "align_logfiles_stability/json_files/parsedlog_255949.json",
         "align_logfiles_stability/json_files/parsedlog_256030.json",
         "align_logfiles_stability/json_files/parsedlog_256145.json",
         "align_logfiles_stability/json_files/parsedlog_256159.json",
         "align_logfiles_stability/json_files/parsedlog_256163.json",
         "align_logfiles_stability/json_files/parsedlog_256272.json",
         "align_logfiles_stability/json_files/parsedlog_256278.json",
         "align_logfiles_stability/json_files/parsedlog_256290.json",
]
legendlabels=[\
              "255949",
              "256030",
              "256145",
              "256159",
              "256163",
              "256272",
              "256278",
              "256290",
]
diff_labels = [\
            "255949-256030",
            "..030-..145",
            "..145-..159",
            "..159-..163",
            "..163-..272",
            "..272-..278",
            "..278-..290",
]

diff_md = [\
            "255949-256030",
            "..030-..145",
            "..145-..159",
            "..159-..163",
]

diff_mu = [\
            "..272-..278",
            "..278-..290",
]

files_md = [\
         "align_logfiles_stability/json_files/parsedlog_255949.json",
         "align_logfiles_stability/json_files/parsedlog_256030.json",
         "align_logfiles_stability/json_files/parsedlog_256145.json",
         "align_logfiles_stability/json_files/parsedlog_256159.json",
         "align_logfiles_stability/json_files/parsedlog_256163.json",
]
legendlabels_md = [\
              "255949",
              "256030",
              "256145",
              "256159",
              "256163",
]

files_mu = [\
         "align_logfiles_stability/json_files/parsedlog_256272.json",
         "align_logfiles_stability/json_files/parsedlog_256278.json",
         "align_logfiles_stability/json_files/parsedlog_256290.json",
]
legendlabels_mu = [\
              "256272",
              "256278",
              "256290",
]

survey_module_positions = 'survey/survey_Modules.yml'

layers = ['T1U', 'T1V', 'T1X1', 'T1X2', 'T2U', 'T2V', 'T2X1', 'T2X2', 'T3U', 'T3V', 'T3X1', 'T3X2']

runs = ["T3UHL0Q0M0", "T3UHL0Q0M1", "T3UHL0Q0M2", "T3UHL0Q0M3", "T3UHL0Q0M4", "T3UHL0Q2M0", "T3UHL0Q2M1", "T3UHL0Q2M2", "T3UHL0Q2M3", "T3UHL0Q2M4", "T3UHL1Q1M0"#\
        , "T3UHL1Q1M1", "T3UHL1Q1M2", "T3UHL1Q1M3", "T3UHL1Q1M4", "T3UHL1Q3M0", "T3UHL1Q3M1", "T3UHL1Q3M2", "T3UHL1Q3M3", "T3UHL1Q3M4"]

# all files
align_outputs=[open_alignment(thisfile) for thisfile in files]
plotted_alignables=[]
for align_block in align_outputs:
    thislist=[]
    for key in align_block.keys():
        if "FT" in key:
            thislist.append(key)
    plotted_alignables.append(thislist)
align_outputs=[convertGlobal(align_block,plotted_alignables[0]) for align_block in align_outputs]

# magnet down
align_outputs_md=[open_alignment(thisfile) for thisfile in files_md]
plotted_alignables_md=[]
for align_block in align_outputs_md:
    thislist_md=[]
    for key in align_block.keys():
        if "FT" in key:
            thislist.append(key)
    plotted_alignables_md.append(thislist)
align_outputs_md=[convertGlobal(align_block,plotted_alignables_md[0]) for align_block in align_outputs_md]

# magnet up
align_outputs_mu=[open_alignment(thisfile) for thisfile in files_mu]
plotted_alignables_mu=[]
for align_block in align_outputs_mu:
    thislist_mu=[]
    for key in align_block.keys():
        if "FT" in key:
            thislist.append(key)
    plotted_alignables_mu.append(thislist)
align_outputs_mu=[convertGlobal(align_block,plotted_alignables_mu[0]) for align_block in align_outputs_mu]

# for all files
tx = get_data(files, 'Tx', align_outputs)
ty = get_data(files, 'Ty', align_outputs)
tz = get_data(files, 'Tz', align_outputs)
nHits = get_data(files, 'nHits', align_outputs)
nTracks = get_data(files, 'nTracks', align_outputs)
# local_chi2 = get_data(files, 'localDeltaChi2', align_outputs)
x_glob = get_data(files, 'x_global', align_outputs)
y_glob = get_data(files, 'y_global', align_outputs)
z_glob = get_data(files, 'z_global', align_outputs)

survey_Tx = get_survey_data(survey_module_positions, 'Tx', 'position')
survey_Ty = get_survey_data(survey_module_positions, 'Ty', 'position')
survey_Tz = get_survey_data(survey_module_positions, 'Tz', 'position')
survey_Rz = get_survey_data(survey_module_positions, 'Rz', 'rotation')

# for magnet down
tx_md = get_data(files_md, 'Tx', align_outputs_md)
ty_md = get_data(files_md, 'Ty', align_outputs_md)
tz_md = get_data(files_md, 'Tz', align_outputs_md)
nHits_md = get_data(files_md, 'nHits', align_outputs_md)
nTracks_md = get_data(files_md, 'nTracks', align_outputs_md)
# local_chi2 = get_data(files, 'localDeltaChi2', align_outputs)
x_glob_md = get_data(files_md, 'x_global', align_outputs_md)
y_glob_md = get_data(files_md, 'y_global', align_outputs_md)
z_glob_md = get_data(files_md, 'z_global', align_outputs_md)

# for magnet up
tx_mu = get_data(files_mu, 'Tx', align_outputs_mu)
ty_mu = get_data(files_mu, 'Ty', align_outputs_mu)
tz_mu = get_data(files_mu, 'Tz', align_outputs_mu)
nHits_mu = get_data(files_mu, 'nHits', align_outputs_mu)
nTracks_mu = get_data(files_mu, 'nTracks', align_outputs_mu)
# local_chi2 = get_data(files, 'localDeltaChi2', align_outputs)
x_glob_mu = get_data(files_mu, 'x_global', align_outputs_mu)
y_glob_mu = get_data(files_mu, 'y_global', align_outputs_mu)
z_glob_mu = get_data(files_mu, 'z_global', align_outputs_mu)

for n in range(12):
    tx_data = tx[n]
    ty_data = ty[n]
    tz_data = tz[n]
    x_g = x_glob[n]
    y_g = y_glob[n]
    z_g = z_glob[n]
    nHits_data = nHits[n]
    nTracks_data = nTracks[n]
    # chi2 = local_chi2[n]
    hits_vs_tracks(nHits_data, nTracks_data, legendlabels, layers[n])
    # plots the frontview quarter plots
    # plot(tx_data, survey_Tx, 'diff_data', legendlabels, 'Tx', layers[n])  # set [] to survey Tx if i want to compare to survey positions
    # plot(tz_data, [], 'diff_data', legendlabels, 'Tz', layers[n])
    plot(tx_data, [], 'diff_data', diff_labels, 'Tx', layers[n])  # set [] to survey Tx if i want to compare to survey positions
    # plot(tz_data, [], 'data_diff', diff_labels, 'Tz', layers[n])

    # plot(nHits_data, [], 'n_Hits', legendlabels, 'nHits', layers[n])
    # plot(nTracks_data, [], 'n_Tracks', legendlabels, 'nTracks', layers[n])
    # plot(chi2, 'chi2', legendlabels, 'localDeltaChi2', layers[n])

    # top view plots
    plot_with_globals(tx, 'global_z_vs_Tx', legendlabels, layers, z_glob, x_glob)
    compare_alignments(tx_data, 'diff_runs', legendlabels, 'Tx', layers[n])
    compare_alignments(nHits_data, 'nHits_diff', legendlabels, 'nHits', layers[n])
    compare_alignments(nTracks_data, 'nTracks_diff', legendlabels, 'nTracks', layers[n])
    # compare_alignments(chi2, 'chi2', z_g, legendlabels, 'localDeltaChi2', layers[n])
    # plotTxTzMapsGlobal(align_outputs, files, legendlabels, layers)

    # now for md only
    tx_data_md = tx_md[n]
    ty_data_md = ty_md[n]
    tz_data_md = tz_md[n]
    x_g_md = x_glob_md[n]
    y_g_md = y_glob_md[n]
    z_g_md = z_glob_md[n]
    plot(tx_data_md, [], 'diff_MD', diff_md, 'Tx', layers[n])  # set [] to survey Tx if i want to compare to survey positions
    plot(tz_data_md, survey_Tx, 'diff_MD', legendlabels_md, 'Tz', layers[n])

    # now for mu only
    tx_data_mu = tx_mu[n]
    ty_data_mu = ty_mu[n]
    tz_data_mu = tz_mu[n]
    x_g_mu = x_glob_mu[n]
    y_g_mu = y_glob_mu[n]
    z_g_mu = z_glob_mu[n]
    diffs = max_module_deviation(tx_data, 'max_deviation', diff_labels, layers[n])
    for i in range(len(diff_labels)):
        print(f'diff_{i}: ', diffs[i])
    plot(tx_data_mu, [], 'diff_MU', diff_mu, 'Tx', layers[n])  # set [] to survey Tx if i want to compare to survey positions
    plot(tz_data_mu, survey_Tz, 'diff_MU', legendlabels_mu, 'Tz', layers[n])


'''
    notes:

    chi2 data seems to be weird -> investigate
'''
