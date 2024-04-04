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
from matplotlib.patches import Rectangle

matplotlib.rcParams['figure.figsize'] = [7.5, 5]

regex_typelabel=re.compile("Q")
regex_amodule=re.compile("dPosXYZ")
regex_rmodule=re.compile("dRotXYZ")
labels=["Tx","Ty","Tz","Rx","Ry","Rz"]
positions=["x_global","y_global","z_global"]
trackInfo=["nTracks","nHits"]
stations = ["T1", "T2", "T3"]

colors = ['black', 'blue', 'red', 'green', 'magenta', 'yellow', 'brown', 'cyan', 'purple']
markers = ['s', 'd', 'v', 'x', '*', 'v', '.', 'p', '1', '2']

halfmodule_length = 4833 / 2  # in mm
halfmodule_width = 536  # in mm
num_modules_per_quarter = 5
n_quarter = 4
n_halves = 2

# change to your own output directories
outname_prefix = '2024_global_alignment/outfiles/'

def global_local_combiner(global_data, local_data, quarter_or_layer, local_or_global):
    '''
        needs data from only one iteration
    '''
    num_files = global_data.shape[0]
    x_Q0_data = [[] for _ in range(num_files)]
    x_Q2_data = [[] for _ in range(num_files)]
    x_Q1_data = [[] for _ in range(num_files)]
    x_Q3_data = [[] for _ in range(num_files)]

    if quarter_or_layer == 'quarter':
        if local_or_global == 'global':
            for i in range(num_files):
                x_Q0_data[i].append(global_data[i][0:5] + local_data[i][0:5])
                x_Q2_data[i].append(global_data[i][5:10] + local_data[i][5:10])
                x_Q1_data[i].append(global_data[i][10:15] + local_data[i][10:15])
                x_Q3_data[i].append(global_data[i][15:20] + local_data[i][15:20])
        if local_or_global == 'local':
            for i in range(num_files):
                x_Q0_data[i].append(local_data[i][0:5])
                x_Q2_data[i].append(local_data[i][5:10])
                x_Q1_data[i].append(local_data[i][10:15])
                x_Q3_data[i].append(local_data[i][15:20])
    if quarter_or_layer == 'layer':
        if local_or_global == 'global':
            for i in range(num_files):
                x_Q0_data[i].append(global_data[i] + local_data[i])
                x_Q2_data[i].append(global_data[i] + local_data[i])
                x_Q1_data[i].append(global_data[i] + local_data[i])
                x_Q3_data[i].append(global_data[i] + local_data[i])
        if local_or_global == 'local':
            for i in range(num_files):
                x_Q0_data[i].append(local_data[i])
                x_Q2_data[i].append(local_data[i])
                x_Q1_data[i].append(local_data[i])
                x_Q3_data[i].append(local_data[i])

    return x_Q0_data, x_Q2_data, x_Q1_data, x_Q3_data

def plot_x_y_constants(nums1, nums2, local1, local2, labels, ID, quarter_or_layer, local_or_global, outname):
    outfiles = 'out_x_y_pos/'
    total_layer_num = 12 # number of layers
    total_num_runs = len(labels)
    # print('nums1:', nums1)
    # print('local1:', local1)
    # x has 4 entries, 1 for each quarter, within these 4 we have as many as the number of input files
    x_Q0, x_Q2, x_Q1, x_Q3 = global_local_combiner(nums1, local1, quarter_or_layer, local_or_global)
    y_Q0, y_Q2, y_Q1, y_Q3 = global_local_combiner(nums2, local2, quarter_or_layer, local_or_global)

    L = ['Q2', 'Q3', 'Q0', 'Q1']
    # print('x_Q2', x_Q2)
    ax = [plt.subplot(2,2,i+1) for i in range(4)]
    plt.figure()
    count = 0
    modules = ["M0", "M1", "M2", "M3", "M4"]
    x = np.linspace(0, 5, 5)

    for a in ax:
        a.text(0.05, 0.9, L[count], transform=a.transAxes, weight="bold")
        plt.sca(a)
        if count == 0: # Q2
            for num in range(total_num_runs): # iterate through files for Q2
                plt.scatter(x_Q2[num][0][::-1], y_Q2[num][0][::-1], color=colors[num], marker=markers[num], s=10)
                for i in range(len(modules)):
                    plt.text(x_Q2[num][0][i], y_Q2[num][0][i], modules[i], fontsize=9)
                plt.ylabel(f'y pos [mm]')
                plt.title(f'{ID} module positions')
                a.invert_yaxis()
                a.xaxis.tick_top()
        if count == 1: # Q3
            for num in range(total_num_runs): # iterate through files for Q2
                plt.scatter(x_Q3[num][0][::-1], y_Q3[num][0][::-1], color=colors[num], marker=markers[num], s=10, label=f'{labels[num]}')
                for i in range(len(modules)):
                    plt.text(x_Q3[num][0][i], y_Q3[num][0][i], modules[i], fontsize=9)
                    a.yaxis.tick_right()
                a.xaxis.tick_top()
                plt.legend(loc='best', fontsize='8')
        if count == 2: # Q0
            for num in range(total_num_runs): # iterate through files for Q2
                plt.scatter(x_Q0[num][0][::-1], y_Q0[num][0][::-1], color=colors[num], marker=markers[num], s=10)
                for i in range(len(modules)):
                    plt.text(x_Q0[num][0][i], y_Q0[num][0][i], modules[i], fontsize=9)
        if count == 3: # Q1
            for num in range(total_num_runs): # iterate through files for Q2
                plt.scatter(x_Q1[num][0][::-1], y_Q1[num][0][::-1], color=colors[num], marker=markers[num], s=10)
                for i in range(len(modules)):
                    plt.text(x_Q1[num][0][i], y_Q1[num][0][i], modules[i], fontsize=9)
                    a.yaxis.tick_right()
                plt.xlabel('x [mm]')
        count += 1
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(f'{outname_prefix}{outfiles}' + f'{outname}' + ID + '.pdf')
    plt.clf()

def plot_with_globals(data_arr, outname, run_labels, layer_names, glob_data1, glob_data2, y_axis):
    outfiles = 'outfiles_vs_global/'
    total_layer_num = len(layer_names)
    total_num_runs = len(run_labels)

    x_data = data_arr
    x_glob = glob_data2
    z_glob = glob_data1
    z_positions = [] # 12 values, 1 for each layer
    for j in range(total_layer_num):
        z_positions.append(z_glob[j][0][0])

    x_means = [[] for _ in range(total_num_runs)]
    if y_axis == 'Tx':
        x_shifted = np.array(x_data) + np.array(x_glob)

        for run in range(total_num_runs):
            for layer in range(total_layer_num):
                x_means[run].append(np.mean(x_shifted[layer][run]))

    if y_axis != 'Tx':
        x_shifted = np.array(x_data)

        for run in range(total_num_runs):
            for layer in range(total_layer_num):
                x_means[run].append(np.mean(x_shifted[layer][run]))

    if y_axis in ["Tx", "Ty", "Tz"]:
        correct_order = [2, 0, 1, 3, 6, 4, 5, 7, 10, 8, 9, 11]
        for runs in range(total_num_runs):
            correct_x_order = [x_means[runs][iter] for iter in correct_order]
            size = 10
            plt.plot(z_positions, x_means[runs], ls='', marker=markers[runs], c=colors[runs], markersize=size, label=f'{run_labels[runs]}')
            if plt.grid(True):
                plt.grid()
            plt.legend(loc='best')
            plt.ylabel(f'mean {y_axis} module position [mm]')
            plt.title(f'mean position of layers in {y_axis} vs. global z position')
            plt.xticks(z_positions, ['T1U', 'T1V', 'T1X1', 'T1X2', 'T2U', 'T2V', 'T2X1', 'T2X2', 'T3U', 'T3V', 'T3X1', 'T3X2'], rotation=45, fontsize=10)
        plt.savefig(f'{outname_prefix}/{outfiles}' + 'all_runs_' + outname + f'{y_axis}.pdf')
        plt.clf()
    else:
        correct_order = [2, 0, 1, 3, 6, 4, 5, 7, 10, 8, 9, 11]
        for runs in range(total_num_runs):
            correct_x_order = [x_means[runs][iter] for iter in correct_order]
            size = 10
            plt.plot(z_positions, x_means[runs], ls='', marker=markers[runs], c=colors[runs], markersize=size, label=f'{run_labels[runs]}')
            if plt.grid(True):
                plt.grid()
            plt.legend(loc='best')
            plt.ylabel(f'mean {y_axis} module position [rad]')
            plt.title(f'mean position of layers in {y_axis} vs. global z position')
            plt.xticks(z_positions, ['T1U', 'T1V', 'T1X1', 'T1X2', 'T2U', 'T2V', 'T2X1', 'T2X2', 'T3U', 'T3V', 'T3X1', 'T3X2'], rotation=45, fontsize=10)
        plt.savefig(f'{outname_prefix}/{outfiles}' + 'all_runs_' + outname + f'{y_axis}.pdf')
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

def get_data(files, DoF, align_output): # , withLongModules=False, withCFrames=False
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
        for j in range(0,len(stations)): # 0 - 3
            for k in range(0,len(layers)): # 0 - 11
                if j==0 and k==0:
                    runs_T1_U[iter_num]=runs_T1
                    runs_T2_U[iter_num]=runs_T2
                    runs_T3_U[iter_num]=runs
                elif j==0 and k==1:
                    for i in range(0,len(runs)):
                        string1 = runs_T1[i]
                        # print(f'string1 = runs_T1[{i}]:', runs_T1[i])
                        string2 = runs_T2[i]
                        string3 = runs[i]
                        runs_T1_V[iter_num].append(string1.replace("T1/U", "T1/V"))
                        # print('runs_T1_V[iter_num]', runs_T1_V[iter_num])
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
            with open(file, 'r') as stream:
                data_loaded = align_output[iter_num]

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

def make_outfiles(files, output_variables): # , withLongModules, withCFrames
    align_outputs=[open_alignment(thisfile) for thisfile in files]
    plotted_alignables=[]
    for align_block in align_outputs:
        thislist=[]
        for key in align_block.keys():
            if "FT" in key:
                if key in long_modules_objects:
                    continue
                elif key in cframe_objects:
                    continue
                else:
                    thislist.append(key)
        plotted_alignables.append(thislist)

    align_outputs=[convertGlobal(align_block,plotted_alignables[0]) for align_block in align_outputs]

    out_vars = []
    for var in output_variables:
        out_vars.append(get_data(files, var, align_outputs)) # , withLongModules, withCFrames
    return out_vars

def meta_constructor(loader, node):
   return loader.construct_mapping(node)

yaml.add_constructor('!alignment', meta_constructor)

# input files

cframe_objects = [\
    'FT/T1/X1U/HL0',
    'FT/T1/X1U/HL1',
    'FT/T1/VX2/HL0',
    'FT/T1/VX2/HL1',
    'FT/T2/X1U/HL0',
    'FT/T2/X1U/HL1',
    'FT/T2/VX2/HL0',
    'FT/T2/VX2/HL1',
    'FT/T3/X1U/HL0',
    'FT/T3/X1U/HL1',
    'FT/T3/VX2/HL0',
    'FT/T3/VX2/HL1'
]

long_modules_objects = [\
'FT/T1/U/HL0/M0',  'FT/T2/U/HL0/M0',  'FT/T3/U/HL0/M0',
'FT/T1/U/HL0/M1',  'FT/T2/U/HL0/M1',  'FT/T3/U/HL0/M1',
'FT/T1/U/HL0/M2',  'FT/T2/U/HL0/M2',  'FT/T3/U/HL0/M2',
'FT/T1/U/HL0/M3',  'FT/T2/U/HL0/M3',  'FT/T3/U/HL0/M3',
'FT/T1/U/HL0/M4',  'FT/T2/U/HL0/M4',  'FT/T3/U/HL0/M4',
'FT/T3/U/HL0/M5',
'FT/T1/U/HL1/M0',  'FT/T2/U/HL1/M0',  'FT/T3/U/HL1/M0',
'FT/T1/U/HL1/M1',  'FT/T2/U/HL1/M1',  'FT/T3/U/HL1/M1',
'FT/T1/U/HL1/M2',  'FT/T2/U/HL1/M2',  'FT/T3/U/HL1/M2',
'FT/T1/U/HL1/M3',  'FT/T2/U/HL1/M3',  'FT/T3/U/HL1/M3',
'FT/T1/U/HL1/M4',  'FT/T2/U/HL1/M4',  'FT/T3/U/HL1/M4',
'FT/T3/U/HL1/M5',
'FT/T1/V/HL0/M0',  'FT/T2/V/HL0/M0',  'FT/T3/V/HL0/M0',
'FT/T1/V/HL0/M1',  'FT/T2/V/HL0/M1',  'FT/T3/V/HL0/M1',
'FT/T1/V/HL0/M2',  'FT/T2/V/HL0/M2',  'FT/T3/V/HL0/M2',
'FT/T1/V/HL0/M3',  'FT/T2/V/HL0/M3',  'FT/T3/V/HL0/M3',
'FT/T1/V/HL0/M4',  'FT/T2/V/HL0/M4',  'FT/T3/V/HL0/M4',
'FT/T3/V/HL0/M5',
'FT/T1/V/HL1/M0',  'FT/T2/V/HL1/M0',  'FT/T3/V/HL1/M0',
'FT/T1/V/HL1/M1',  'FT/T2/V/HL1/M1',  'FT/T3/V/HL1/M1',
'FT/T1/V/HL1/M2',  'FT/T2/V/HL1/M2',  'FT/T3/V/HL1/M2',
'FT/T1/V/HL1/M3',  'FT/T2/V/HL1/M3',  'FT/T3/V/HL1/M3',
'FT/T1/V/HL1/M4',  'FT/T2/V/HL1/M4',  'FT/T3/V/HL1/M4',
'FT/T3/V/HL1/M5',
'FT/T1/X1/HL0/M0', 'FT/T2/X1/HL0/M0', 'FT/T3/X1/HL0/M0',
'FT/T1/X1/HL0/M1', 'FT/T2/X1/HL0/M1', 'FT/T3/X1/HL0/M1',
'FT/T1/X1/HL0/M2', 'FT/T2/X1/HL0/M2', 'FT/T3/X1/HL0/M2',
'FT/T1/X1/HL0/M3', 'FT/T2/X1/HL0/M3', 'FT/T3/X1/HL0/M3',
'FT/T1/X1/HL0/M4', 'FT/T2/X1/HL0/M4', 'FT/T3/X1/HL0/M4',
'FT/T3/X1/HL0/M5',
'FT/T1/X1/HL1/M0', 'FT/T2/X1/HL1/M0', 'FT/T3/X1/HL1/M0',
'FT/T1/X1/HL1/M1', 'FT/T2/X1/HL1/M1', 'FT/T3/X1/HL1/M1',
'FT/T1/X1/HL1/M2', 'FT/T2/X1/HL1/M2', 'FT/T3/X1/HL1/M2',
'FT/T1/X1/HL1/M3', 'FT/T2/X1/HL1/M3', 'FT/T3/X1/HL1/M3',
'FT/T1/X1/HL1/M4', 'FT/T2/X1/HL1/M4', 'FT/T3/X1/HL1/M4',
'FT/T3/X1/HL1/M5',
'FT/T1/X2/HL0/M0', 'FT/T2/X2/HL0/M0', 'FT/T3/X2/HL0/M0',
'FT/T1/X2/HL0/M1', 'FT/T2/X2/HL0/M1', 'FT/T3/X2/HL0/M1',
'FT/T1/X2/HL0/M2', 'FT/T2/X2/HL0/M2', 'FT/T3/X2/HL0/M2',
'FT/T1/X2/HL0/M3', 'FT/T2/X2/HL0/M3', 'FT/T3/X2/HL0/M3',
'FT/T1/X2/HL0/M4', 'FT/T2/X2/HL0/M4', 'FT/T3/X2/HL0/M4',
'FT/T3/X2/HL0/M5',
'FT/T1/X2/HL1/M0', 'FT/T2/X2/HL1/M0', 'FT/T3/X2/HL1/M0',
'FT/T1/X2/HL1/M1', 'FT/T2/X2/HL1/M1', 'FT/T3/X2/HL1/M1',
'FT/T1/X2/HL1/M2', 'FT/T2/X2/HL1/M2', 'FT/T3/X2/HL1/M2',
'FT/T1/X2/HL1/M3', 'FT/T2/X2/HL1/M3', 'FT/T3/X2/HL1/M3',
'FT/T1/X2/HL1/M4', 'FT/T2/X2/HL1/M4', 'FT/T3/X2/HL1/M4',
'FT/T3/X2/HL1/M5',
]

layers = ['T1U', 'T1V', 'T1X1', 'T1X2', 'T2U', 'T2V', 'T2X1', 'T2X2', 'T3U', 'T3V', 'T3X1', 'T3X2']
vars = ['Tx', 'Ty', 'Tz', 'Rx', 'Ry', 'Rz', 'nHits', 'nTracks', 'x_global', 'y_global', 'z_global']

'''
    how to run and what can be changed:
    1. run "parse_alignlog_to_json.py" to make parsedlog.json file from alignlog.txt
    2. make a "files" and a "legendlabels" array and fill it with the json files you want to run and add labels for them
    3. copy the block of 10 lines with Tx_v1, Ty_v1, ... and give it new names for your new run
    4. same for the lines inside the for-loop over the layers so the extraction of variables in per layer and global is set
    5. at the bottom where you call the function, you can just copy the lines and use your newly set names
'''

path_mar3 = "2024_global_alignment/03-15"
path_mar4 = "2024_global_alignment/03-18"

files_v1 = [\
    f"{path_mar4}/0_01/v1_2/parsedlog.json",
    f"{path_mar4}/0_01/v1_3/parsedlog.json",
    f"{path_mar3}/0_01/v1_2/parsedlog.json",
]
legendlabels_v1 = [\
    '0.01 VELO Rx, SciFi TxRz',
    '0.01 VELO Rx, SciFi TxRxRz',
    '0.01 VELO Rx, SciFi TxRxRyRz',
]

# here add a new line to combine the align outputs for a correct formatting
align_outputs_v1 = make_outfiles(files_v1, vars)

# all these need to be copied with new names for your new align_ouputs
# the dataformat here has all information for all layers
Tx_v1      = align_outputs_v1[0]
Ty_v1      = align_outputs_v1[1]
Tz_v1      = align_outputs_v1[2]
Rx_v1      = align_outputs_v1[3]
Ry_v1      = align_outputs_v1[4]
Rz_v1      = align_outputs_v1[5]
nHits_v1   = align_outputs_v1[6]
nTracks_v1 = align_outputs_v1[7]
x_glob_v1  = align_outputs_v1[8]
y_glob_v1  = align_outputs_v1[9]
z_glob_v1  = align_outputs_v1[10]

# this here takes information for a each individual layer so we can make layer plots
for n in range(12):
    Tx_data_v1 = Tx_v1[n]
    Ty_data_v1 = Ty_v1[n]
    Tz_data_v1 = Tz_v1[n]
    Rx_data_v1 = Rx_v1[n]
    Ry_data_v1 = Ry_v1[n]
    Rz_data_v1 = Rz_v1[n]
    x_g_v1 = x_glob_v1[n]
    y_g_v1 = y_glob_v1[n]
    z_g_v1 = z_glob_v1[n]

    # this plots the constants in x vs y for each layer and for each quarter
    plot_x_y_constants(x_g_v1, y_g_v1, Tx_data_v1, Ty_data_v1, legendlabels_v1, layers[n], 'quarter', 'global', 'v1_x_vs_z') # for global coordinates in x
    plot_x_y_constants(x_g_v1, y_g_v1, Tx_data_v1, Ty_data_v1, legendlabels_v1, layers[n], 'quarter', 'local', 'v1_x_vs_y_local') # for local coordinates in x

# plot for global z vs layer mean in the given degree of freedom
plot_with_globals(Tx_v1, 'v1_glob_z_vs_local_', legendlabels_v1, layers, z_glob_v1, x_glob_v1, 'Tx')
plot_with_globals(Ty_v1, 'v1_glob_z_vs_local_', legendlabels_v1, layers, z_glob_v1, x_glob_v1, 'Ty')
plot_with_globals(Tz_v1, 'v1_glob_z_vs_local_', legendlabels_v1, layers, z_glob_v1, x_glob_v1, 'Tz')
plot_with_globals(Rx_v1, 'v1_glob_z_vs_local_', legendlabels_v1, layers, z_glob_v1, x_glob_v1, 'Rx')
plot_with_globals(Ry_v1, 'v1_glob_z_vs_local_', legendlabels_v1, layers, z_glob_v1, x_glob_v1, 'Ry')
plot_with_globals(Rz_v1, 'v1_glob_z_vs_local_', legendlabels_v1, layers, z_glob_v1, x_glob_v1, 'Rz')
