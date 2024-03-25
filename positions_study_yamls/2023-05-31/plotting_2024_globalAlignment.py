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

def make_edges_plot(nums1, nums2, local1, local2, labels, ID, quarter_or_layer, local_or_global, x_rot, outname, filenumbers='all'):
    outfiles = 'joint_touching/'
    total_layer_num = 12 # number of layers
    total_num_runs = len(labels)

    # x has 4 entries, 1 for each quarter, within these 4 we have as many as the number of input files
    x_Q0, x_Q2, x_Q1, x_Q3 = global_local_combiner(nums1, local1, quarter_or_layer, local_or_global)
    y_Q0, y_Q2, y_Q1, y_Q3 = global_local_combiner(nums2, local2, quarter_or_layer, local_or_global)
    # only the local numbers per layer
    y1, y2, y3, y4 = global_local_combiner(nums2, local2, 'layer', 'local') # all y are the same because per layer, but have to define it this way
    # rx rotation
    rx_data = [[] for _ in range(total_num_runs)]
    for i in range(total_num_runs):
        rx_data[i].append(x_rot[i])

    L = ['Q2', 'Q3', 'Q0', 'Q1']
    len_long_module = 2417.5 # mm, dont use this
    global_joint = [0, -1212.75, 0] # x, y, z
    # for a test do one half layer -> test Q0 top edges and Q2 bottom edges
    dim_modules = len(y_Q2[0][0])
    x = np.linspace(0, 10, 10)
    s1 = set(['T1X1', 'T1X2', 'T2X1', 'T2X2', 'T3X1', 'T3X2']) # X1, X2 layers, +- 1213 mm
    s2 = set(['T1U', 'T1V', 'T2U', 'T2V', 'T3U', 'T3V']) #  U,  V layers, +- 1208 mm
    fig, ax2 = plt.subplots()
    if filenumbers == 'all':
        for num in range(total_num_runs):
            if ID in s2:
                # print('U and V layers')
                stereo_angle = np.cos(np.deg2rad(5))
                y_data = [y_Q2[num][0][i] / stereo_angle for i in range(dim_modules)]
            else:
                y_data = [y_Q2[num][0][i] for i in range(dim_modules)]
            y_final = [] # 20 entries, 1 for each module
            edge_factor = [1,1,1,1,1,-1,-1,-1,-1,-1,1,1,1,1,1,-1,-1,-1,-1,-1]
            for j in range(dim_modules):
                y_final.append(y_data[j])# * np.cos(rx_data[num][0][j]))
                y_final[j] *= edge_factor[j]
            top_idx = [5, 6, 7, 8, 9, 15, 16, 17, 18, 19]
            bot_idx = [0, 1, 2, 3, 4, 10, 11, 12, 13, 14]
            y_top = [y_final[i] for i in top_idx]
            y_bot = [y_final[i] for i in bot_idx]

            top_blc = [] # blc = bottom left corner for top half modules
            bottom_blc = [] # for bottomhalf modules
            for i in range(len(y_top)):
                if i < 5:
                    top_blc.append([-halfmodule_width * (i+1), y_top[i], 0])
                    bottom_blc.append([-halfmodule_width * (i+1), y_bot[i], 0])
                else:
                    top_blc.append([halfmodule_width * (i-5), y_top[i], 0])
                    bottom_blc.append([halfmodule_width * (i-5), y_bot[i], 0])
            y_top_min = min(y_top)
            y_top_max = max(y_top)
            y_bottom_min = min(y_bot)
            y_bottom_max = max(y_bot)
            # print('y_top_min', y_top_min)
            # print('y_top_max', y_top_max)
            # print('y_bottom_min', y_bottom_min)
            # print('y_bottom_max', y_bottom_max)
            for module in range(len(top_blc)): # top
                ax2.add_patch(Rectangle((top_blc[module][0], top_blc[module][1]), halfmodule_width, 0, edgecolor=colors[num], linewidth=1, linestyle="--", fill=False))
                ax2.set_xlim(-2700, 2700)
                ax2.set_ylim(y_top_max, y_top_min)
            for module in range(len(bottom_blc)): # bottom
                ax2.add_patch(Rectangle((bottom_blc[module][0], bottom_blc[module][1]), halfmodule_width, 0, edgecolor=colors[num], linewidth=1, linestyle=":" , fill=False))
                ax2.set_ylim(y_bottom_max, y_bottom_min)
            plt.title('module fitting at the joint')
            plt.xlabel('global module position in x [mm]')
            plt.ylabel('global module position in y [mm]')
            plt.savefig(f'{outname_prefix}{outfiles}' + f'{outname}_run{num}_{filenumbers}_{ID}.pdf')
        plt.clf()
    if filenumbers == 'individual':
        for num in range(total_num_runs):
            if ID in s2:
                # print('U and V layers')
                stereo_angle = np.cos(np.deg2rad(5))
                y_data = [y_Q2[num][0][i] / stereo_angle for i in range(dim_modules)]
            else:
                y_data = [y_Q2[num][0][i] for i in range(dim_modules)]
            y_final = [] # 20 entries, 1 for each module
            edge_factor = [1,1,1,1,1,-1,-1,-1,-1,-1,1,1,1,1,1,-1,-1,-1,-1,-1]
            for j in range(dim_modules):
                y_final.append(y_data[j] * np.cos(rx_data[num][0][j]))
                y_final[j] *= edge_factor[j]
            top_idx = [5, 6, 7, 8, 9, 15, 16, 17, 18, 19]
            bot_idx = [0, 1, 2, 3, 4, 10, 11, 12, 13, 14]
            y_top = [y_final[i] for i in top_idx]
            y_bot = [y_final[i] for i in bot_idx]

            top_blc = [] # blc = bottom left corner for top half modules
            bottom_blc = [] # for bottomhalf modules
            for i in range(len(y_top)):
                if i < 5:
                    top_blc.append([-halfmodule_width * (i+1), y_top[i], 0])
                    bottom_blc.append([-halfmodule_width * (i+1), y_bot[i], 0])
                else:
                    top_blc.append([halfmodule_width * (i-5), y_top[i], 0])
                    bottom_blc.append([halfmodule_width * (i-5), y_bot[i], 0])
            y_top_min = min(y_top)
            y_top_max = max(y_top)
            y_bottom_min = min(y_bot)
            y_bottom_max = max(y_bot)
            # print('y_top_min', y_top_min)
            # print('y_top_max', y_top_max)
            # print('y_bottom_min', y_bottom_min)
            # print('y_bottom_max', y_bottom_max)
            for module in range(len(top_blc)): # top
                ax2.add_patch(Rectangle((top_blc[module][0], top_blc[module][1]), halfmodule_width, 0, edgecolor=colors[num], linewidth=1, linestyle="--", fill=False))
                ax2.set_xlim(-2700, 2700)
                ax2.set_ylim(y_top_max, y_top_min)
            for module in range(len(bottom_blc)): # bottom
                ax2.add_patch(Rectangle((bottom_blc[module][0], bottom_blc[module][1]), halfmodule_width, 0, edgecolor=colors[num], linewidth=1, linestyle=":" , fill=False))
                ax2.set_xlim(-2700, 2700)
                ax2.set_ylim(y_bottom_max, y_bottom_min)
            plt.title('module fitting at the joint')
            plt.xlabel('global module position in x [mm]')
            plt.ylabel('global module position in y [mm]')
            plt.savefig(f'{outname_prefix}{outfiles}' + f'{outname}_{filenumbers}_{ID}_file{num}.pdf')
        plt.clf()

def check_module_edges(nums1, nums2, local1, local2, labels, ID, quarter_or_layer, local_or_global, x_rot, outname):
    '''
    instead of plotting x vs y -> check if the top and bottom module edges touch at the joint at 0 -1212.75 0
    '''
    outfiles = 'out_x_y_pos/'
    total_layer_num = 12 # number of layers
    total_num_runs = len(labels)

    # x has 4 entries, 1 for each quarter, within these 4 we have as many as the number of input files
    x_Q0, x_Q2, x_Q1, x_Q3 = global_local_combiner(nums1, local1, quarter_or_layer, local_or_global)
    y_Q0, y_Q2, y_Q1, y_Q3 = global_local_combiner(nums2, local2, quarter_or_layer, local_or_global)
    # only the local numbers per layer
    y1, y2, y3, y4 = global_local_combiner(nums2, local2, 'layer', 'local') # all y are the same because per layer, but have to define it this way
    # rx rotation
    rx_data = [[] for _ in range(total_num_runs)]
    for i in range(total_num_runs):
        rx_data[i].append(x_rot[i])

    L = ['Q2', 'Q3', 'Q0', 'Q1']
    len_long_module = 2417.5 # mm, dont use this
    '''
        procedure:
        1.) if U, V: y_global *= cos(deg2rad(5))
        1.5) else just use y_global
        2.) erg1 = y_global +- y_local
        3.) shift from Rx rotation: erg2 = np.sqrt(erg1**2 - (abs(erg1) * np.sin(Rx))**2)
    '''
    global_joint = [0, -1212.75, 0] # x, y, z
    # for a test do one half layer -> test Q0 top edges and Q2 bottom edges
    dim_modules = len(y_Q2[0][0])
    x = np.linspace(0, 10, 10)
    s1 = set(['T1X1', 'T1X2', 'T2X1', 'T2X2', 'T3X1', 'T3X2']) # X1, X2 layers, +- 1213 mm
    s2 = set(['T1U', 'T1V', 'T2U', 'T2V', 'T3U', 'T3V']) #  U,  V layers, +- 1208 mm
    for num in range(total_num_runs):
        if ID in s2:
            # print('U and V layers')
            stereo_angle = np.cos(np.deg2rad(5))
            y_data = [y_Q2[num][0][i] / stereo_angle for i in range(dim_modules)]
            # print('y_data for U or V layer:', y_data)
        else:
            y_data = [y_Q2[num][0][i] for i in range(dim_modules)]
            # print('y_data for X1 or X2 layer:', y_data)
        # print('rx data from run[num]:', rx_data[num][0])
        # shift from Rx on top
        y_final = [] # 20 entries, 1 for each module
        edge_factor = [1,1,1,1,1,-1,-1,-1,-1,-1,1,1,1,1,1,-1,-1,-1,-1,-1]
        for j in range(dim_modules):
            y_final.append(y_data[j] * np.cos(rx_data[num][0][j]))
            y_final[j] *= edge_factor[j]
        # print('###################################')
        # print('y_final', y_final)
        top_idx = [5, 6, 7, 8, 9, 15, 16, 17, 18, 19]
        bot_idx = [0, 1, 2, 3, 4, 10, 11, 12, 13, 14]
        y_top = [y_final[i] for i in top_idx]
        y_bot = [y_final[i] for i in bot_idx]
        # print('y_top', y_top)
        # print('y_bot', y_bot)
        # make_edges_plot(y_top, y_bot, ID, num)
        plt.scatter(x, y_top, color=colors[num], marker='.', label=f'{labels[num]}')
        plt.scatter(x, y_bot, color=colors[num], marker='x')
        # plt.hlines(global_joint[1], x[0], x[9], 'red')
        # plt.text(x[1], -1212.85, '. : top quarters')
        # plt.text(x[1], -1212.86, 'x : bottom quarters')
        plt.grid()
        plt.legend()
        if len(x_Q2[num][0]) == 5:
            plt.xticks(x, ["Q0M0", "Q0M1", "Q0M2", "Q0M3", "Q0M4"])
        else:
            plt.xticks(x, ["HL0/M0", "HL0/M1", "HL0/M2", "HL0/M3", "HL0/M4", "HL1/M0", "HL1/M1", "HL1/M2", "HL1/M3", "HL1/M4"], rotation=45)
        plt.ylabel(f'global module edge')
        plt.xlabel('modules')
        plt.title(f'global position of the touching edges')
        plt.savefig(f'{outname_prefix}{outfiles}' + f'{outname}' + ID + '.pdf')
    plt.clf()

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

def glob_vs_glob(glob_data1, glob_data2, dof1, dof2, outname, labels):
    run_labels = labels
    num_files = glob_data1[0].shape[0]
    n_layers = 12
    layer_ticks = ['T1U', 'T1V', 'T1X1', 'T1X2', 'T2U', 'T2V', 'T2X1', 'T2X2', 'T3U', 'T3V', 'T3X1', 'T3X2']
    outfiles = 'outfiles_vs_global/'
    # glob 1 is y coordinate, glob 2 is z coordinate
    z_positions = []
    for j in range(n_layers):
        z_positions.append(glob_data2[j][0][0])
    correct_order = [2, 0, 1, 3, 6, 4, 5, 7, 10, 8, 9, 11]
    for runs in range(num_files):
        y_positions = []
        for j in range(n_layers):
            y_positions.append(glob_data1[j][0][j])

        # print('y_positions', y_positions)
        correct_x_order = [y_positions[iter] for iter in correct_order]
        # print('z_positions', z_positions)
        plt.plot(z_positions, y_positions, ls='', marker=markers[runs], c=colors[runs], markersize=11-runs, label=f'{run_labels[runs]}')
        if runs == 0:
            for i in range(n_layers):
                plt.text(z_positions[i], 1100-(runs*50), layer_ticks[i], fontsize=9)
        if plt.grid(True):
            plt.grid()
        plt.legend(loc='best')
        plt.xlabel(f'{dof2} [mm]')
        plt.ylabel(f'{dof1} [mm]')
        plt.title(f'{dof1} vs {dof2}')
    plt.savefig(f'{outname_prefix}/{outfiles}' + outname + '.pdf')
    plt.clf()

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
    # change from json  order (U, V, X1, X2) to physical (X1, U, V, X2)
#    if y_axis == 'Ty':
    if y_axis in ["Tx", "Ty", "Tz"]:    
        correct_order = [2, 0, 1, 3, 6, 4, 5, 7, 10, 8, 9, 11]
        for runs in range(total_num_runs):
            correct_x_order = [x_means[runs][iter] for iter in correct_order]
            size = 10
            plt.plot(z_positions, x_means[runs], ls='', marker=markers[runs], c=colors[runs], markersize=size, label=f'{run_labels[runs]}')
            # if runs == 0:
            #     for i in range(total_layer_num):
            #         plt.text(x_means[runs][i], z_positions[i]+((i/20) * (-1)**i), layers[i], fontsize=9)
            if plt.grid(True):
                plt.grid()
            plt.legend(loc='best')
            plt.ylabel(f'mean {y_axis} module position [mm]')
            plt.title(f'mean position of layers in {y_axis} vs. global z position')
            plt.xticks(z_positions, ['T1U', 'T1V', 'T1X1', 'T1X2', 'T2U', 'T2V', 'T2X1', 'T2X2', 'T3U', 'T3V', 'T3X1', 'T3X2'], rotation=45, fontsize=10)
        plt.savefig(f'{outname_prefix}/{outfiles}' + 'all_runs_' + outname + f'{y_axis}.pdf')
        plt.clf()
    # if y_axis == 'Rx' or y_axis == 'Tx':
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
    if y_axis == 'Tz':
        correct_order = [2, 0, 1, 3, 6, 4, 5, 7, 10, 8, 9, 11]
        for runs in range(total_num_runs):
            correct_x_order = [x_means[runs][iter] for iter in correct_order]
            size = 10
            if runs == 3:
                size = 11
            if runs == 5:
                size = 8
            plt.plot(x_means[runs], z_positions, ls='', marker=markers[runs], c=colors[runs], markersize=size, label=f'{run_labels[runs]}')
            if plt.grid(True):
                plt.grid()
            plt.legend(loc='best')
            plt.ylabel(f'global position of SciFi Layers')
            plt.xlabel('local mean Tz movement of Layers')
            plt.title(f'mean position of layers in {y_axis} vs. global z position')
            plt.yticks(z_positions, ['T1U', 'T1V', 'T1X1', 'T1X2', 'T2U', 'T2V', 'T2X1', 'T2X2', 'T3U', 'T3V', 'T3X1', 'T3X2'], rotation=45, fontsize=10)
        plt.savefig(f'{outname_prefix}/{outfiles}' + 'all_runs_' + outname + f'{y_axis}.pdf')
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

    path = '2024_global_alignment'
    # if files[0] == f'{path}/01-24/retest_pink_giulia_conditions/Iter14/parsedlog.json' or files[0] == f'{path}/01-24/retest_pink_giulia_no_cframes/Iter14/parsedlog.json' or files[0] == f'{path}/01-24/retest_pink_without_velo/Iter14/parsedlog.json' or files[0] == f"{path}/01-26/pink_giulia/last_iter/parsedlog.json" or files[0] == f"{path}/01-29/diff_survey_giulia_pink/Iter29/parsedlog.json" or files[0] == f"{path}/02-07/fhtl_FT_VP/Iter4/parsedlog.json" or files[0] == f"{path}/02-09/flth_cframes/parsedlog.json" or files[0] == f"{path}/02-16/backward_vp/Iter3/parsedlog.json" or files[0] == f"{path}/02-28/v1_1/Iter9/parsedlog.json":
    # if withLongModules == True and withCFrames == False:
        # print('withLongModules == True and withCFrames == False')
        # runs_T1 = ['FT/T1/U/HL0/M0', 'FT/T1/U/HL0/M1', 'FT/T1/U/HL0/M2', 'FT/T1/U/HL0/M3', 'FT/T1/U/HL0/M4',
                #    'FT/T1/U/HL1/M0', 'FT/T1/U/HL1/M1', 'FT/T1/U/HL1/M2', 'FT/T1/U/HL1/M3', 'FT/T1/U/HL1/M4']
        # runs_T2 = ['FT/T2/U/HL0/M0', 'FT/T2/U/HL0/M1', 'FT/T2/U/HL0/M2', 'FT/T2/U/HL0/M3', 'FT/T2/U/HL0/M4',
                #    'FT/T2/U/HL1/M0', 'FT/T2/U/HL1/M1', 'FT/T2/U/HL1/M2', 'FT/T2/U/HL1/M3', 'FT/T2/U/HL1/M4']
        # runs = ['FT/T3/U/HL0/M0', 'FT/T3/U/HL0/M1', 'FT/T3/U/HL0/M2', 'FT/T3/U/HL0/M3', 'FT/T3/U/HL0/M4', # , 'FT/T3/U/HL0/M5'
                # 'FT/T3/U/HL1/M0', 'FT/T3/U/HL1/M1', 'FT/T3/U/HL1/M2', 'FT/T3/U/HL1/M3', 'FT/T3/U/HL1/M4'] # , 'FT/T3/U/HL0/M5'
    # if withLongModules == False and withCFrames == False:
        # print('withLongModules == False and withCFrames == False')
        # runs_T1 = ["FT/T1/U/HL0/Q0/M0", "FT/T1/U/HL0/Q0/M1", "FT/T1/U/HL0/Q0/M2", "FT/T1/U/HL0/Q0/M3", "FT/T1/U/HL0/Q0/M4",
                #    "FT/T1/U/HL0/Q2/M0", "FT/T1/U/HL0/Q2/M1", "FT/T1/U/HL0/Q2/M2", "FT/T1/U/HL0/Q2/M3", "FT/T1/U/HL0/Q2/M4",
                #    "FT/T1/U/HL1/Q1/M0", "FT/T1/U/HL1/Q1/M1", "FT/T1/U/HL1/Q1/M2", "FT/T1/U/HL1/Q1/M3", "FT/T1/U/HL1/Q1/M4",
                #    "FT/T1/U/HL1/Q3/M0", "FT/T1/U/HL1/Q3/M1", "FT/T1/U/HL1/Q3/M2", "FT/T1/U/HL1/Q3/M3", "FT/T1/U/HL1/Q3/M4"]
    # 
        # runs_T2 = ["FT/T2/U/HL0/Q0/M0", "FT/T2/U/HL0/Q0/M1", "FT/T2/U/HL0/Q0/M2", "FT/T2/U/HL0/Q0/M3", "FT/T2/U/HL0/Q0/M4",
                #    "FT/T2/U/HL0/Q2/M0", "FT/T2/U/HL0/Q2/M1", "FT/T2/U/HL0/Q2/M2", "FT/T2/U/HL0/Q2/M3", "FT/T2/U/HL0/Q2/M4",
                #    "FT/T2/U/HL1/Q1/M0", "FT/T2/U/HL1/Q1/M1", "FT/T2/U/HL1/Q1/M2", "FT/T2/U/HL1/Q1/M3", "FT/T2/U/HL1/Q1/M4",
                #    "FT/T2/U/HL1/Q3/M0", "FT/T2/U/HL1/Q3/M1", "FT/T2/U/HL1/Q3/M2", "FT/T2/U/HL1/Q3/M3", "FT/T2/U/HL1/Q3/M4"]
    # 
        # runs = ["FT/T3/U/HL0/Q0/M0", "FT/T3/U/HL0/Q0/M1", "FT/T3/U/HL0/Q0/M2", "FT/T3/U/HL0/Q0/M3", "FT/T3/U/HL0/Q0/M4",
                # "FT/T3/U/HL0/Q2/M0", "FT/T3/U/HL0/Q2/M1", "FT/T3/U/HL0/Q2/M2", "FT/T3/U/HL0/Q2/M3", "FT/T3/U/HL0/Q2/M4",
                # "FT/T3/U/HL1/Q1/M0", "FT/T3/U/HL1/Q1/M1", "FT/T3/U/HL1/Q1/M2", "FT/T3/U/HL1/Q1/M3", "FT/T3/U/HL1/Q1/M4",
                # "FT/T3/U/HL1/Q3/M0", "FT/T3/U/HL1/Q3/M1", "FT/T3/U/HL1/Q3/M2", "FT/T3/U/HL1/Q3/M3", "FT/T3/U/HL1/Q3/M4"]
    # if withLongModules == False and withCFrames == True:
        # print('withLongModules == False and withCFrames == True')
        # runs_T1 = ['FT/T1/X1U/HL0', "FT/T1/U/HL0/Q0/M0", "FT/T1/U/HL0/Q0/M1", "FT/T1/U/HL0/Q0/M2", "FT/T1/U/HL0/Q0/M3", "FT/T1/U/HL0/Q0/M4",
                #    'FT/T1/X1U/HL1', "FT/T1/U/HL0/Q2/M0", "FT/T1/U/HL0/Q2/M1", "FT/T1/U/HL0/Q2/M2", "FT/T1/U/HL0/Q2/M3", "FT/T1/U/HL0/Q2/M4",
                #    'FT/T1/VX2/HL0', "FT/T1/U/HL1/Q1/M0", "FT/T1/U/HL1/Q1/M1", "FT/T1/U/HL1/Q1/M2", "FT/T1/U/HL1/Q1/M3", "FT/T1/U/HL1/Q1/M4",
                #    'FT/T1/VX2/HL1', "FT/T1/U/HL1/Q3/M0", "FT/T1/U/HL1/Q3/M1", "FT/T1/U/HL1/Q3/M2", "FT/T1/U/HL1/Q3/M3", "FT/T1/U/HL1/Q3/M4"]
    # 
        # runs_T2 = ['FT/T2/X1U/HL0', "FT/T2/U/HL0/Q0/M0", "FT/T2/U/HL0/Q0/M1", "FT/T2/U/HL0/Q0/M2", "FT/T2/U/HL0/Q0/M3", "FT/T2/U/HL0/Q0/M4",
                #    'FT/T2/X1U/HL1', "FT/T2/U/HL0/Q2/M0", "FT/T2/U/HL0/Q2/M1", "FT/T2/U/HL0/Q2/M2", "FT/T2/U/HL0/Q2/M3", "FT/T2/U/HL0/Q2/M4",
                #    'FT/T2/VX2/HL0', "FT/T2/U/HL1/Q1/M0", "FT/T2/U/HL1/Q1/M1", "FT/T2/U/HL1/Q1/M2", "FT/T2/U/HL1/Q1/M3", "FT/T2/U/HL1/Q1/M4",
                #    'FT/T2/VX2/HL1', "FT/T2/U/HL1/Q3/M0", "FT/T2/U/HL1/Q3/M1", "FT/T2/U/HL1/Q3/M2", "FT/T2/U/HL1/Q3/M3", "FT/T2/U/HL1/Q3/M4"]
    # 
        # runs = ['FT/T3/X1U/HL0', "FT/T3/U/HL0/Q0/M0", "FT/T3/U/HL0/Q0/M1", "FT/T3/U/HL0/Q0/M2", "FT/T3/U/HL0/Q0/M3", "FT/T3/U/HL0/Q0/M4",
                # 'FT/T3/X1U/HL1', "FT/T3/U/HL0/Q2/M0", "FT/T3/U/HL0/Q2/M1", "FT/T3/U/HL0/Q2/M2", "FT/T3/U/HL0/Q2/M3", "FT/T3/U/HL0/Q2/M4",
                # 'FT/T3/VX2/HL0', "FT/T3/U/HL1/Q1/M0", "FT/T3/U/HL1/Q1/M1", "FT/T3/U/HL1/Q1/M2", "FT/T3/U/HL1/Q1/M3", "FT/T3/U/HL1/Q1/M4",
                # 'FT/T3/VX2/HL1', "FT/T3/U/HL1/Q3/M0", "FT/T3/U/HL1/Q3/M1", "FT/T3/U/HL1/Q3/M2", "FT/T3/U/HL1/Q3/M3", "FT/T3/U/HL1/Q3/M4"]
    # if withLongModules == True and withCFrames == True:
        # print('withLongModules == True and withCFrames == True')
        # runs_T1 = ['FT/T1/X1U/HL0', 'FT/T1/X1U/HL1', 'FT/T1/VX2/HL0', 'FT/T1/VX2/HL1',
                #    'FT/T1/U/HL0/M0', 'FT/T1/U/HL0/M1', 'FT/T1/U/HL0/M2', 'FT/T1/U/HL0/M3', 'FT/T1/U/HL0/M4',
                #    'FT/T1/U/HL1/M0', 'FT/T1/U/HL1/M1', 'FT/T1/U/HL1/M2', 'FT/T1/U/HL1/M3', 'FT/T1/U/HL1/M4',
                #    "FT/T1/U/HL0/Q0/M0", "FT/T1/U/HL0/Q0/M1", "FT/T1/U/HL0/Q0/M2", "FT/T1/U/HL0/Q0/M3", "FT/T1/U/HL0/Q0/M4",
                #    "FT/T1/U/HL0/Q2/M0", "FT/T1/U/HL0/Q2/M1", "FT/T1/U/HL0/Q2/M2", "FT/T1/U/HL0/Q2/M3", "FT/T1/U/HL0/Q2/M4",
                #    "FT/T1/U/HL1/Q1/M0", "FT/T1/U/HL1/Q1/M1", "FT/T1/U/HL1/Q1/M2", "FT/T1/U/HL1/Q1/M3", "FT/T1/U/HL1/Q1/M4",
                #    "FT/T1/U/HL1/Q3/M0", "FT/T1/U/HL1/Q3/M1", "FT/T1/U/HL1/Q3/M2", "FT/T1/U/HL1/Q3/M3", "FT/T1/U/HL1/Q3/M4"]
    # 
        # runs_T2 = ['FT/T2/X1U/HL0', 'FT/T2/X1U/HL1', 'FT/T2/VX2/HL0', 'FT/T2/VX2/HL1',
                #    'FT/T2/U/HL0/M0', 'FT/T2/U/HL0/M1', 'FT/T2/U/HL0/M2', 'FT/T2/U/HL0/M3', 'FT/T2/U/HL0/M4',
                #    'FT/T2/U/HL1/M0', 'FT/T2/U/HL1/M1', 'FT/T2/U/HL1/M2', 'FT/T2/U/HL1/M3', 'FT/T2/U/HL1/M4',
                #    "FT/T2/U/HL0/Q0/M0", "FT/T2/U/HL0/Q0/M1", "FT/T2/U/HL0/Q0/M2", "FT/T2/U/HL0/Q0/M3", "FT/T2/U/HL0/Q0/M4",
                #    "FT/T2/U/HL0/Q2/M0", "FT/T2/U/HL0/Q2/M1", "FT/T2/U/HL0/Q2/M2", "FT/T2/U/HL0/Q2/M3", "FT/T2/U/HL0/Q2/M4",
                #    "FT/T2/U/HL1/Q1/M0", "FT/T2/U/HL1/Q1/M1", "FT/T2/U/HL1/Q1/M2", "FT/T2/U/HL1/Q1/M3", "FT/T2/U/HL1/Q1/M4",
                #    "FT/T2/U/HL1/Q3/M0", "FT/T2/U/HL1/Q3/M1", "FT/T2/U/HL1/Q3/M2", "FT/T2/U/HL1/Q3/M3", "FT/T2/U/HL1/Q3/M4"]
    # 
        # runs = ['FT/T3/X1U/HL0', 'FT/T3/X1U/HL1', 'FT/T3/VX2/HL0', 'FT/T3/VX2/HL1',
                # 'FT/T3/U/HL0/M0', 'FT/T3/U/HL0/M1', 'FT/T3/U/HL0/M2', 'FT/T3/U/HL0/M3', 'FT/T3/U/HL0/M4',
                # 'FT/T3/U/HL1/M0', 'FT/T3/U/HL1/M1', 'FT/T3/U/HL1/M2', 'FT/T3/U/HL1/M3', 'FT/T3/U/HL1/M4',
                # "FT/T3/U/HL0/Q0/M0", "FT/T3/U/HL0/Q0/M1", "FT/T3/U/HL0/Q0/M2", "FT/T3/U/HL0/Q0/M3", "FT/T3/U/HL0/Q0/M4",
                # "FT/T3/U/HL0/Q2/M0", "FT/T3/U/HL0/Q2/M1", "FT/T3/U/HL0/Q2/M2", "FT/T3/U/HL0/Q2/M3", "FT/T3/U/HL0/Q2/M4",
                # "FT/T3/U/HL1/Q1/M0", "FT/T3/U/HL1/Q1/M1", "FT/T3/U/HL1/Q1/M2", "FT/T3/U/HL1/Q1/M3", "FT/T3/U/HL1/Q1/M4",
                # "FT/T3/U/HL1/Q3/M0", "FT/T3/U/HL1/Q3/M1", "FT/T3/U/HL1/Q3/M2", "FT/T3/U/HL1/Q3/M3", "FT/T3/U/HL1/Q3/M4"]
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
        # print('runs_T1_U', runs_T1_U)
        # print('runs_T1_U', runs_T1_V)
        for i in range(0,len(runs)):
            # what is the json file used for
            with open(file, 'r') as stream:  # why do i do this???
                data_loaded = align_output[iter_num]
                # print(data_loaded)
                # print(data_loaded[runs_T1_U[iter_num][i]].keys())
                # print(deg, data_loaded[runs_T1_U[iter_num][i]].values())
                # print(data_loaded[runs_T1_U['FT/T1/U/HL0/M0']])
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
                # print(key)
                # thislist.append(key)
                if key in long_modules_objects:
                    continue
                elif key in cframe_objects:
                    continue
                else:
                    # print(key)
                    thislist.append(key)
        plotted_alignables.append(thislist)
    # if withLongModules == False:
    align_outputs=[convertGlobal(align_block,plotted_alignables[0]) for align_block in align_outputs]
    # else:
    #     align_output=deepcopy(align_block)
    #     for alignable in plotted_alignables:
    #         if "Quarter0" in alignable or "Quarter1" in alignable or "Q0" in alignable or "Q1" in alignable:
    #             for label in ["Tx","Rx","Ry","Ty"]:
    #                 if label in align_output[alignable].keys():
    #                     align_output[alignable][label]=[-value for value in align_output[alignable][label]]
    #         if "X1" in alignable or "V" in alignable:
    #             if "Quarter0" in alignable or "Quarter1" in alignable or "Q0" in alignable or "Q1" in alignable:
    #                 # minus sign on x and Rx and z and Rz for all iterations
    #                 for label in ["Tx","Rx","Rz","Tz"]:
    #                     if label in align_output[alignable].keys():
    #                         align_output[alignable][label]=[-value for value in align_output[alignable][label]]
    #             else:
    #                 continue
    out_vars = []
    for var in output_variables:
        out_vars.append(get_data(files, var, align_outputs)) # , withLongModules, withCFrames
    # return align_outputs
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

path = '2024_global_alignment/01-24'
files_v1 = [\
        f"{path}/retest_pink_giulia_conditions/Iter14/parsedlog.json",
        # f"{path}/retest_pink_giulia_no_cframes/Iter14/parsedlog.json",
        f"{path}/retest_pink_without_velo/Iter14/parsedlog.json",
]
legendlabels_v1=[\
              "v1,pink,giulia condition,VELO+SciFi",
              # "v1,pink,giulia condition,VELO+SciFi, no CFrame option",
              "v1,pink,giulia condition,SciFi",
]

path1 = '2024_global_alignment/01-26'
files_v2 = [\
        f"{path1}/pink_giulia/last_iter/parsedlog.json",
        # f"{path1}/pink_giulia_no_cframe_survey/last_iter/parsedlog.json",
        f"{path1}/pink_giulia_no_velo/last_iter/parsedlog.json",
]
legendlabels_v2=[\
              "v2,pink,giulia condition,VELO+SciFi",
              # "v2,pink,giulia condition,VELO+SciFi, no CFrame option",
              "v2,pink,giulia condition,SciFi",
]

path2 = '2024_global_alignment/01-29'
files_v3 = [\
        f"{path2}/diff_survey_giulia_pink/Iter29/parsedlog.json",
        # f"{path2}/diff_survey_pink_no_cframe/Iter29/parsedlog.json",
        f"{path2}/diff_data_no_velo/Iter29/parsedlog.json",
]
legendlabels_v3=[\
              "v3,pink,giulia condition,VELO+SciFi",
              # "v3,pink,giulia condition,VELO+SciFi, no CFrame option",
              "v3,pink,giulia condition,SciFi",
]

# all giualia with velo + SciFi
files_both_det = [\
    f"{path}/retest_pink_giulia_conditions/Iter14/parsedlog.json",
    f"{path1}/pink_giulia/last_iter/parsedlog.json",
    f"{path2}/diff_survey_giulia_pink/Iter29/parsedlog.json", # not converged
]
legendlabels_both=[\
    "v1,pink,giulia condition,VELO+SciFi",
    "v2,pink,giulia condition,VELO+SciFi",
    "v3,pink,giulia condition,VELO+SciFi",
]

files_scifi = [\
    f"{path}/retest_pink_without_velo/Iter14/parsedlog.json",
    f"{path1}/pink_giulia_no_velo/last_iter/parsedlog.json",
    f"{path2}/diff_data_no_velo/Iter29/parsedlog.json",
]
legendlabels_scifi=[\
    "v1,pink,giulia condition,SciFi",
    "v2,pink,giulia condition,SciFi",
    "v3,pink,giulia condition,SciFi",
]

# first half then long modules aligned until convergence
# and other way round
# TODO: being able to use json files which have only long modules
# -> maybe make an input option?
path3 = '2024_global_alignment/02-07'
files_hm_first = [\
    f"{path3}/fhtl_FT_VP/Iter4/parsedlog.json",
    f"{path3}/flth_FT_VP/Iter9/parsedlog.json",
]

legendlabels_hm_first = [\
    "SciFi+VELO, HM only",
    "Scifi+Velo, half+long mod"
]

# 02-09 fixes for biljana meeting on 14th feb
path4 = '2024_global_alignment/02-09'
path5 = "2024_global_alignment/02-13"
path6 = "2024_global_alignment/02-16"
path7 = "retest_uncertainty/input_txt/loose_particles/global_alignment_files/2023-11-07/v1_veloRx"
path8 = "2024_global_alignment/01-31-debug_convergence/"
files_fixes = [\
    f"{path4}/flth_cframes/parsedlog.json",
    # f"{path4}/flth_lagrange_T2/parsedlog.json",
    # f"{path4}/flth_lagrange_T2_total/parsedlog.json",
    f"{path5}/outer_inner/parsedlog.json",
    # f"{path6}/only_T1/parsedlog.json",
    f"{path6}/backward_vp/Iter3/parsedlog.json",
    f"{path7}/parsedlog.json",
    # f"{path8}/parsedlog.json"
    # f"{path6}/backward_vp/Iter11/parsedlog.json"
]
legendlabels_fixes = [\
    "c-frames: long+half",
    # "c-frames: T2 lagrange",
    # "c-frames: lagrange total pos",
    "c-frames: outer-inner lagrange T2",
    # "c-frames: outer-inner lagrange T1",
    "c-frames: bw vp tracks",
    "SF HM + fullVP Rx + VPHalves TxTyTzRy",
    # "SciFi only, HM TxRz"
    # "c-frames: bw vp_tracks, iter11",
]

# here: new input from wouter with Rx tests
path_rx = "2024_global_alignment/02-28"
path_mar = "2024_global_alignment/03-13"
path_mar2 = "2024_global_alignment/03-14"
path_mar3 = "2024_global_alignment/03-15"
path_mar4 = "2024_global_alignment/03-18"
files_Rx_test = [\
#    f"{path_rx}/v1_2/Iter9/parsedlog.json",
    # f"{path_rx}/v1_3/Iter9/parsedlog.json",
#    f"{path_mar3}/v1_2/parsedlog.json",
    # f"{path_mar3}/v1_3/parsedlog.json",
    # f"{path_mar}/v2_3/parsedlog.json",
    f"{path_mar2}/v3_3/parsedlog.json",
    f"{path_mar4}/0_01/v1_2/parsedlog.json",
    f"{path_mar4}/0_01/v1_3/parsedlog.json",
    f"{path_mar3}/0_01/v1_2/parsedlog.json",
]

legendlabels_rx = [\
#    'v1_2',
    # 'v1_3',
#    '0.001 VELO Rx, no SciFi Rx',
    # '0.001 VELO Rx, SciFi TxRxRz',
    # '0.0001, VELO Rx, Giulia_starting conds',
    '0.0001, VELO Rx, all0 conds',
    '0.01 VELO Rx, SciFi TxRz',
    '0.01 VELO Rx, SciFi TxRxRz',
    '0.01 VELO Rx, SciFi TxRxRyRz',
]

# Halfmodules TxRz(Rx forv2_3 and v3_3)
# iD = in Depth
files_iD = [\
#    f"{path_mar}/v2_2/parsedlog.json",
    f"{path_mar}/v2_2/parsedlog.json",
#    f"{path_mar2}/v3_2/parsedlog.json",
    f"{path_mar2}/v3_2/parsedlog.json",
#    f"{path_mar2}/0_0002/v1_2/parsedlog.json",
#    f"{path_mar2}/0_0002/v1_3/parsedlog.json",
#    f"{path_mar3}/v1_2/parsedlog.json",
    # f"{path_mar3}/v1_3/parsedlog.json",
    # f"{path_mar3}/0_01/v1_2/parsedlog.json",
]

legendlabels_iD = [\
#    '0.0001, VELO Rx, Giulia_starting conds',
    '0.0001, VELO Rx, SciFi tag conds',
#    '0.0001, VELO Rx, all0 conds',
    '0.0001, VELO Rx, all0 conds',
#    '0.0002 VELO Rx, no SciFi Rx',
#    '0.0002 VELO Rx, with SciFi Rx',
#    '0.001 VELO Rx, no SciFi Rx',
    # '0.001 VELO Rx, with SciFi Rx',
    # '0.01 VELO Rx, with SciFi Ry',
]

survey_module_positions = 'survey/survey_Modules.yml'

layers = ['T1U', 'T1V', 'T1X1', 'T1X2', 'T2U', 'T2V', 'T2X1', 'T2X2', 'T3U', 'T3V', 'T3X1', 'T3X2']

runs = ["T3UHL0Q0M0", "T3UHL0Q0M1", "T3UHL0Q0M2", "T3UHL0Q0M3", "T3UHL0Q0M4", "T3UHL0Q2M0", "T3UHL0Q2M1", "T3UHL0Q2M2", "T3UHL0Q2M3", "T3UHL0Q2M4", "T3UHL1Q1M0"#\
        , "T3UHL1Q1M1", "T3UHL1Q1M2", "T3UHL1Q1M3", "T3UHL1Q1M4", "T3UHL1Q3M0", "T3UHL1Q3M1", "T3UHL1Q3M2", "T3UHL1Q3M3", "T3UHL1Q3M4"]

plotting_variables = ['Tx', 'Ty', 'Tz', 'nHits', 'nTracks', 'x_global', 'y_global', 'z_global']
red_blue_vars = ['Tx', 'Ty', 'Tz', 'Rx', 'Ry', 'Rz', 'nHits', 'nTracks', 'x_global', 'y_global', 'z_global']

align_outputs_v1 = make_outfiles(files_v1, red_blue_vars) # , withLongModules=False, withCFrames=False
align_outputs_v2 = make_outfiles(files_v2, red_blue_vars)
align_outputs_v3 = make_outfiles(files_v3, red_blue_vars)
align_outputs_both = make_outfiles(files_both_det, red_blue_vars)
align_outputs_scifi = make_outfiles(files_scifi, red_blue_vars)
align_outputs_hmf = make_outfiles(files_hm_first, red_blue_vars)
align_outputs_fixes = make_outfiles(files_fixes, red_blue_vars)
align_outputs_rx = make_outfiles(files_Rx_test, red_blue_vars)
align_outputs_iD = make_outfiles(files_iD, red_blue_vars)

# inDepth
Tx_iD      = align_outputs_iD[0]
Ty_iD      = align_outputs_iD[1]
Tz_iD      = align_outputs_iD[2]
Rx_iD      = align_outputs_iD[3]
Ry_iD      = align_outputs_iD[4]
Rz_iD      = align_outputs_iD[5]
nHits_iD   = align_outputs_iD[6]
nTracks_iD = align_outputs_iD[7]
x_glob_iD  = align_outputs_iD[8]
y_glob_iD  = align_outputs_iD[9]
z_glob_iD  = align_outputs_iD[10]

# rx test
Tx_rx      = align_outputs_rx[0]
Ty_rx      = align_outputs_rx[1]
Tz_rx      = align_outputs_rx[2]
Rx_rx      = align_outputs_rx[3]
Ry_rx      = align_outputs_rx[4]
Rz_rx      = align_outputs_rx[5]
nHits_rx   = align_outputs_rx[6]
nTracks_rx = align_outputs_rx[7]
x_glob_rx  = align_outputs_rx[8]
y_glob_rx  = align_outputs_rx[9]
z_glob_rx  = align_outputs_rx[10]

# fixes
Tx_fix      = align_outputs_fixes[0]
Ty_fix      = align_outputs_fixes[1]
Tz_fix      = align_outputs_fixes[2]
Rx_fix      = align_outputs_fixes[3]
Ry_fix      = align_outputs_fixes[4]
Rz_fix      = align_outputs_fixes[5]
nHits_fix   = align_outputs_fixes[6]
nTracks_fix = align_outputs_fixes[7]
x_glob_fix  = align_outputs_fixes[8]
y_glob_fix  = align_outputs_fixes[9]
z_glob_fix  = align_outputs_fixes[10]

# hmf = half modules first
Tx_hmf      = align_outputs_hmf[0]
Ty_hmf      = align_outputs_hmf[1]
Tz_hmf      = align_outputs_hmf[2]
Rx_hmf      = align_outputs_hmf[3]
Ry_hmf      = align_outputs_hmf[4]
Rz_hmf      = align_outputs_hmf[5]
nHits_hmf   = align_outputs_hmf[6]
nTracks_hmf = align_outputs_hmf[7]
x_glob_hmf  = align_outputs_hmf[8]
y_glob_hmf  = align_outputs_hmf[9]
z_glob_hmf  = align_outputs_hmf[10]

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

Tx_v2      = align_outputs_v2[0]
Ty_v2      = align_outputs_v2[1]
Tz_v2      = align_outputs_v2[2]
Rx_v2      = align_outputs_v2[3]
Ry_v2      = align_outputs_v2[4]
Rz_v2      = align_outputs_v2[5]
nHits_v2   = align_outputs_v2[6]
nTracks_v2 = align_outputs_v2[7]
x_glob_v2  = align_outputs_v2[8]
y_glob_v2  = align_outputs_v2[9]
z_glob_v2  = align_outputs_v2[10]

Tx_v3      = align_outputs_v3[0]
Ty_v3      = align_outputs_v3[1]
Tz_v3      = align_outputs_v3[2]
Rx_v3      = align_outputs_v3[3]
Ry_v3      = align_outputs_v3[4]
Rz_v3      = align_outputs_v3[5]
nHits_v3   = align_outputs_v3[6]
nTracks_v3 = align_outputs_v3[7]
x_glob_v3  = align_outputs_v3[8]
y_glob_v3  = align_outputs_v3[9]
z_glob_v3  = align_outputs_v3[10]

Tx_both      = align_outputs_both[0]
Ty_both      = align_outputs_both[1]
Tz_both      = align_outputs_both[2]
Rx_both      = align_outputs_both[3]
Ry_both      = align_outputs_both[4]
Rz_both      = align_outputs_both[5]
nHits_both   = align_outputs_both[6]
nTracks_both = align_outputs_both[7]
x_glob_both  = align_outputs_both[8]
y_glob_both  = align_outputs_both[9]
z_glob_both  = align_outputs_both[10]

Tx_scifi      = align_outputs_scifi[0]
Ty_scifi      = align_outputs_scifi[1]
Tz_scifi      = align_outputs_scifi[2]
Rx_scifi      = align_outputs_scifi[3]
Ry_scifi      = align_outputs_scifi[4]
Rz_scifi      = align_outputs_scifi[5]
nHits_scifi   = align_outputs_scifi[6]
nTracks_scifi = align_outputs_scifi[7]
x_glob_scifi  = align_outputs_scifi[8]
y_glob_scifi  = align_outputs_scifi[9]
z_glob_scifi  = align_outputs_scifi[10]

for n in range(12):
    Tx_data_iD = Tx_iD[n]
    Ty_data_iD = Ty_iD[n]
    Tz_data_iD = Tz_iD[n]
    Rx_data_iD = Rx_iD[n]
    Ry_data_iD = Ry_iD[n]
    Rz_data_iD = Rz_iD[n]
    x_g_iD = x_glob_iD[n]
    y_g_iD = y_glob_iD[n]
    z_g_iD = z_glob_iD[n]

    Tx_data_rx = Tx_rx[n]
    Ty_data_rx = Ty_rx[n]
    Tz_data_rx = Tz_rx[n]
    Rx_data_rx = Rx_rx[n]
    Ry_data_rx = Ry_rx[n]
    Rz_data_rx = Rz_rx[n]
    x_g_rx = x_glob_rx[n]
    y_g_rx = y_glob_rx[n]
    z_g_rx = z_glob_rx[n]

    Tx_data_fix = Tx_fix[n]
    Ty_data_fix = Ty_fix[n]
    Tz_data_fix = Tz_fix[n]
    Rx_data_fix = Rx_fix[n]
    Ry_data_fix = Ry_fix[n]
    Rz_data_fix = Rz_fix[n]
    x_g_fix = x_glob_fix[n]
    y_g_fix = y_glob_fix[n]
    z_g_fix = z_glob_fix[n]

    Tx_data_hmf = Tx_hmf[n]
    Ty_data_hmf = Ty_hmf[n]
    Tz_data_hmf = Tz_hmf[n]
    Rx_data_hmf = Rx_hmf[n]
    Ry_data_hmf = Ry_hmf[n]
    Rz_data_hmf = Rz_hmf[n]
    x_g_hmf = x_glob_hmf[n]
    y_g_hmf = y_glob_hmf[n]
    z_g_hmf = z_glob_hmf[n]

    Tx_data_v1 = Tx_v1[n]
    Ty_data_v1 = Ty_v1[n]
    Tz_data_v1 = Tz_v1[n]
    Rx_data_v1 = Rx_v1[n]
    Ry_data_v1 = Ry_v1[n]
    Rz_data_v1 = Rz_v1[n]
    x_g_v1 = x_glob_v1[n]
    y_g_v1 = y_glob_v1[n]
    z_g_v1 = z_glob_v1[n]

    Tx_data_v2 = Tx_v2[n]
    Ty_data_v2 = Ty_v2[n]
    Tz_data_v2 = Tz_v2[n]
    Rx_data_v2 = Rx_v2[n]
    Ry_data_v2 = Ry_v2[n]
    Rz_data_v2 = Rz_v2[n]
    x_g_v2 = x_glob_v2[n]
    y_g_v2 = y_glob_v2[n]
    z_g_v2 = z_glob_v2[n]

    Tx_data_v3 = Tx_v3[n]
    Ty_data_v3 = Ty_v3[n]
    Tz_data_v3 = Tz_v3[n]
    Rx_data_v3 = Rx_v3[n]
    Ry_data_v3 = Ry_v3[n]
    Rz_data_v3 = Rz_v3[n]
    x_g_v3 = x_glob_v3[n]
    y_g_v3 = y_glob_v3[n]
    z_g_v3 = z_glob_v3[n]

    Tx_data_both = Tx_both[n]
    Ty_data_both = Ty_both[n]
    Tz_data_both = Tz_both[n]
    Rx_data_both = Rx_both[n]
    Ry_data_both = Ry_both[n]
    Rz_data_both = Rz_both[n]
    x_g_both = x_glob_both[n]
    y_g_both = y_glob_both[n]
    z_g_both = z_glob_both[n]

    Tx_data_scifi = Tx_scifi[n]
    Ty_data_scifi = Ty_scifi[n]
    Tz_data_scifi = Tz_scifi[n]
    Rx_data_scifi = Rx_scifi[n]
    Ry_data_scifi = Ry_scifi[n]
    Rz_data_scifi = Rz_scifi[n]
    x_g_scifi = x_glob_scifi[n]
    y_g_scifi = y_glob_scifi[n]
    z_g_scifi = z_glob_scifi[n]

    plot_x_y_constants(x_g_v1, y_g_v1, Tx_data_v1, Ty_data_v1, legendlabels_v1, layers[n], 'quarter', 'global', 'v1_x_vs_z')
    plot_x_y_constants(x_g_v2, y_g_v2, Tx_data_v2, Ty_data_v2, legendlabels_v2, layers[n], 'quarter', 'global', 'v2_x_vs_z')
    plot_x_y_constants(x_g_v3, y_g_v3, Tx_data_v3, Ty_data_v3, legendlabels_v3, layers[n], 'quarter', 'global', 'v3_x_vs_z')
    plot_x_y_constants(x_g_both, y_g_both, Tx_data_both, Ty_data_both, legendlabels_both, layers[n], 'quarter', 'global', 'both_x_vs_z')
    plot_x_y_constants(x_g_scifi, y_g_scifi, Tx_data_scifi, Ty_data_scifi, legendlabels_scifi, layers[n], 'quarter', 'global', 'scifi_x_vs_z')
    plot_x_y_constants(x_g_hmf, y_g_hmf, Tx_data_hmf, Ty_data_hmf, legendlabels_hm_first, layers[n], 'quarter', 'global', 'hmf_x_vs_z')
    plot_x_y_constants(x_g_fix, y_g_fix, Tx_data_fix, Ty_data_fix, legendlabels_fixes, layers[n], 'quarter', 'global', 'fix_x_vs_z')
    plot_x_y_constants(x_g_rx, y_g_rx, Tx_data_rx, Ty_data_rx, legendlabels_rx, layers[n], 'quarter', 'global', 'rx_x_vs_z')
    plot_x_y_constants(x_g_iD, y_g_iD, Tx_data_iD, Ty_data_iD, legendlabels_iD, layers[n], 'quarter', 'global', 'iD_x_vs_z')

    plot_x_y_constants(x_g_v1, y_g_v1, Tx_data_v1, Ty_data_v1, legendlabels_v1, layers[n], 'quarter', 'local', 'v1_x_vs_y_local')
    plot_x_y_constants(x_g_v2, y_g_v2, Tx_data_v2, Ty_data_v2, legendlabels_v2, layers[n], 'quarter', 'local', 'v2_x_vs_y_local')
    plot_x_y_constants(x_g_v3, y_g_v3, Tx_data_v3, Ty_data_v3, legendlabels_v3, layers[n], 'quarter', 'local', 'v3_x_vs_y_local')
    plot_x_y_constants(x_g_both, y_g_both, Tx_data_both, Ty_data_both, legendlabels_both, layers[n], 'quarter', 'local', 'both_x_vs_z_local')
    plot_x_y_constants(x_g_scifi, y_g_scifi, Tx_data_scifi, Ty_data_scifi, legendlabels_scifi, layers[n], 'quarter', 'local', 'scifi_x_vs_z_local')
    plot_x_y_constants(x_g_hmf, y_g_hmf, Tx_data_hmf, Ty_data_hmf, legendlabels_hm_first, layers[n], 'quarter', 'local', 'hmf_x_vs_z_local')
    plot_x_y_constants(x_g_fix, y_g_fix, Tx_data_fix, Ty_data_fix, legendlabels_fixes, layers[n], 'quarter', 'local', 'fix_x_vs_z_local')
    plot_x_y_constants(x_g_rx, y_g_rx, Tx_data_rx, Ty_data_rx, legendlabels_rx, layers[n], 'quarter', 'local', 'rx_x_vs_z_local')
    plot_x_y_constants(x_g_iD, y_g_iD, Tx_data_iD, Ty_data_iD, legendlabels_iD, layers[n], 'quarter', 'local', 'iD_x_vs_z_local')

    check_module_edges(x_g_v1, y_g_v1, Tx_data_v1, Ty_data_v1, legendlabels_v1, layers[n], 'layer', 'global', Rx_data_v1, 'v1_global_align')
    check_module_edges(x_g_v2, y_g_v2, Tx_data_v2, Ty_data_v2, legendlabels_v2, layers[n], 'layer', 'global', Rx_data_v2, 'v2_global_align')
    check_module_edges(x_g_v3, y_g_v3, Tx_data_v3, Ty_data_v3, legendlabels_v3, layers[n], 'layer', 'global', Rx_data_v3, 'v3_global_align')
    check_module_edges(x_g_both, y_g_both, Tx_data_both, Ty_data_both, legendlabels_both, layers[n], 'layer', 'global', Rx_data_both, 'both_global_align')
    check_module_edges(x_g_scifi, y_g_scifi, Tx_data_scifi, Ty_data_scifi, legendlabels_scifi, layers[n], 'layer', 'global', Rx_data_scifi, 'scifi_global_align')
    check_module_edges(x_g_hmf, y_g_hmf, Tx_data_hmf, Ty_data_hmf, legendlabels_hm_first, layers[n], 'layer', 'global', Rx_data_hmf, 'hmf_global_align')
    check_module_edges(x_g_fix, y_g_fix, Tx_data_fix, Ty_data_fix, legendlabels_fixes, layers[n], 'layer', 'global', Rx_data_fix, 'fix_global_align')
    check_module_edges(x_g_rx, y_g_rx, Tx_data_rx, Ty_data_rx, legendlabels_rx, layers[n], 'layer', 'global', Rx_data_rx, 'rx_global_align')
    check_module_edges(x_g_iD, y_g_iD, Tx_data_iD, Ty_data_iD, legendlabels_iD, layers[n], 'layer', 'global', Rx_data_iD, 'iD_global_align')

    # do it for each individual datafile
    # all files
#    make_edges_plot(x_g_v1, y_g_v1, Tx_data_v1, Ty_data_v1, legendlabels_v1, layers[n], 'layer', 'global', Rx_data_v1, 'v1_global_align', 'all')
    # individual
#    make_edges_plot(x_g_v1, y_g_v1, Tx_data_v1, Ty_data_v1, legendlabels_v1, layers[n], 'layer', 'global', Rx_data_v1, 'v1_global_align', 'individual')
    # global plot for TxTzRxRz for 10 mu vs TxRz
    # make_3D_constants(tx_compxy, ty_compxy, tz_compxy, glob_x_compxy, glob_y_compxy, glob_z_compxy, labels_xy, layers[n])
plot_with_globals(Tx_iD, 'iD_glob_z_vs_local_', legendlabels_iD, layers, z_glob_iD, x_glob_iD, 'Tx')
plot_with_globals(Ty_iD, 'iD_glob_z_vs_local_', legendlabels_iD, layers, z_glob_iD, x_glob_iD, 'Ty')
plot_with_globals(Tz_iD, 'iD_glob_z_vs_local_', legendlabels_iD, layers, z_glob_iD, x_glob_iD, 'Tz')
plot_with_globals(Rx_iD, 'iD_glob_z_vs_local_', legendlabels_iD, layers, z_glob_iD, x_glob_iD, 'Rx')
plot_with_globals(Ry_iD, 'iD_glob_z_vs_local_', legendlabels_iD, layers, z_glob_iD, x_glob_iD, 'Ry')
plot_with_globals(Rz_iD, 'iD_glob_z_vs_local_', legendlabels_iD, layers, z_glob_iD, x_glob_iD, 'Rz')

plot_with_globals(Tx_rx, 'rx_glob_z_vs_local_', legendlabels_rx, layers, z_glob_rx, x_glob_rx, 'Tx')
plot_with_globals(Ty_rx, 'rx_glob_z_vs_local_', legendlabels_rx, layers, z_glob_rx, x_glob_rx, 'Ty')
plot_with_globals(Tz_rx, 'rx_glob_z_vs_local_', legendlabels_rx, layers, z_glob_rx, x_glob_rx, 'Tz')
plot_with_globals(Rx_rx, 'rx_glob_z_vs_local_', legendlabels_rx, layers, z_glob_rx, x_glob_rx, 'Rx')
plot_with_globals(Ry_rx, 'rx_glob_z_vs_local_', legendlabels_rx, layers, z_glob_rx, x_glob_rx, 'Ry')
plot_with_globals(Rz_rx, 'rx_glob_z_vs_local_', legendlabels_rx, layers, z_glob_rx, x_glob_rx, 'Rz')

plot_with_globals(Tx_fix, 'fix_glob_z_vs_local_', legendlabels_fixes, layers, z_glob_fix, x_glob_fix, 'Tx')
plot_with_globals(Ty_fix, 'fix_glob_z_vs_local_', legendlabels_fixes, layers, z_glob_fix, x_glob_fix, 'Ty')
plot_with_globals(Tz_fix, 'fix_glob_z_vs_local_', legendlabels_fixes, layers, z_glob_fix, x_glob_fix, 'Tz')
plot_with_globals(Rx_fix, 'fix_glob_z_vs_local_', legendlabels_fixes, layers, z_glob_fix, x_glob_fix, 'Rx')
plot_with_globals(Ry_fix, 'fix_glob_z_vs_local_', legendlabels_fixes, layers, z_glob_fix, x_glob_fix, 'Ry')
plot_with_globals(Rz_fix, 'fix_glob_z_vs_local_', legendlabels_fixes, layers, z_glob_fix, x_glob_fix, 'Rz')

plot_with_globals(Tx_hmf, 'hmf_glob_z_vs_local_', legendlabels_hm_first, layers, z_glob_hmf, x_glob_hmf, 'Tx')
plot_with_globals(Ty_hmf, 'hmf_glob_z_vs_local_', legendlabels_hm_first, layers, z_glob_hmf, x_glob_hmf, 'Ty')
plot_with_globals(Tz_hmf, 'hmf_glob_z_vs_local_', legendlabels_hm_first, layers, z_glob_hmf, x_glob_hmf, 'Tz')
plot_with_globals(Rx_hmf, 'hmf_glob_z_vs_local_', legendlabels_hm_first, layers, z_glob_hmf, x_glob_hmf, 'Rx')
plot_with_globals(Ry_hmf, 'hmf_glob_z_vs_local_', legendlabels_hm_first, layers, z_glob_hmf, x_glob_hmf, 'Ry')
plot_with_globals(Rz_hmf, 'hmf_glob_z_vs_local_', legendlabels_hm_first, layers, z_glob_hmf, x_glob_hmf, 'Rz')

plot_with_globals(Tx_v1, 'v1_glob_z_vs_local_', legendlabels_v1, layers, z_glob_v1, x_glob_v1, 'Tx')
plot_with_globals(Ty_v1, 'v1_glob_z_vs_local_', legendlabels_v1, layers, z_glob_v1, x_glob_v1, 'Ty')
plot_with_globals(Tz_v1, 'v1_glob_z_vs_local_', legendlabels_v1, layers, z_glob_v1, x_glob_v1, 'Tz')
plot_with_globals(Rx_v1, 'v1_glob_z_vs_local_', legendlabels_v1, layers, z_glob_v1, x_glob_v1, 'Rx')
plot_with_globals(Ry_v1, 'v1_glob_z_vs_local_', legendlabels_v1, layers, z_glob_v1, x_glob_v1, 'Ry')
plot_with_globals(Rz_v1, 'v1_glob_z_vs_local_', legendlabels_v1, layers, z_glob_v1, x_glob_v1, 'Rz')

plot_with_globals(Tx_v2, 'v2_glob_z_vs_local_', legendlabels_v2, layers, z_glob_v2, x_glob_v2, 'Tx')
plot_with_globals(Ty_v2, 'v2_glob_z_vs_local_', legendlabels_v2, layers, z_glob_v2, x_glob_v2, 'Ty')
plot_with_globals(Tz_v2, 'v2_glob_z_vs_local_', legendlabels_v2, layers, z_glob_v2, x_glob_v2, 'Tz')
plot_with_globals(Rx_v2, 'v2_glob_z_vs_local_', legendlabels_v2, layers, z_glob_v2, x_glob_v2, 'Rx')
plot_with_globals(Ry_v2, 'v2_glob_z_vs_local_', legendlabels_v2, layers, z_glob_v2, x_glob_v2, 'Ry')
plot_with_globals(Rz_v2, 'v2_glob_z_vs_local_', legendlabels_v2, layers, z_glob_v2, x_glob_v2, 'Rz')

plot_with_globals(Tx_v3, 'v3_glob_z_vs_local_', legendlabels_v3, layers, z_glob_v3, x_glob_v3, 'Tx')
plot_with_globals(Ty_v3, 'v3_glob_z_vs_local_', legendlabels_v3, layers, z_glob_v3, x_glob_v3, 'Ty')
plot_with_globals(Tz_v3, 'v3_glob_z_vs_local_', legendlabels_v3, layers, z_glob_v3, x_glob_v3, 'Tz')
plot_with_globals(Rx_v3, 'v3_glob_z_vs_local_', legendlabels_v3, layers, z_glob_v3, x_glob_v3, 'Rx')
plot_with_globals(Ry_v3, 'v3_glob_z_vs_local_', legendlabels_v3, layers, z_glob_v3, x_glob_v3, 'Ry')
plot_with_globals(Rz_v3, 'v3_glob_z_vs_local_', legendlabels_v3, layers, z_glob_v3, x_glob_v3, 'Rz')

plot_with_globals(Tx_both, 'both_glob_z_vs_local_', legendlabels_both, layers, z_glob_both, x_glob_both, 'Tx')
plot_with_globals(Ty_both, 'both_glob_z_vs_local_', legendlabels_both, layers, z_glob_both, x_glob_both, 'Ty')
plot_with_globals(Tz_both, 'both_glob_z_vs_local_', legendlabels_both, layers, z_glob_both, x_glob_both, 'Tz')
plot_with_globals(Rx_both, 'both_glob_z_vs_local_', legendlabels_both, layers, z_glob_both, x_glob_both, 'Rx')
plot_with_globals(Ry_both, 'both_glob_z_vs_local_', legendlabels_both, layers, z_glob_both, x_glob_both, 'Ry')
plot_with_globals(Rz_both, 'both_glob_z_vs_local_', legendlabels_both, layers, z_glob_both, x_glob_both, 'Rz')

plot_with_globals(Tx_scifi, 'scifi_glob_z_vs_local_', legendlabels_scifi, layers, z_glob_scifi, x_glob_scifi, 'Tx')
plot_with_globals(Ty_scifi, 'scifi_glob_z_vs_local_', legendlabels_scifi, layers, z_glob_scifi, x_glob_scifi, 'Ty')
plot_with_globals(Tz_scifi, 'scifi_glob_z_vs_local_', legendlabels_scifi, layers, z_glob_scifi, x_glob_scifi, 'Tz')
plot_with_globals(Rx_scifi, 'scifi_glob_z_vs_local_', legendlabels_scifi, layers, z_glob_scifi, x_glob_scifi, 'Rx')
plot_with_globals(Ry_scifi, 'scifi_glob_z_vs_local_', legendlabels_scifi, layers, z_glob_scifi, x_glob_scifi, 'Ry')
plot_with_globals(Rz_scifi, 'scifi_glob_z_vs_local_', legendlabels_scifi, layers, z_glob_scifi, x_glob_scifi, 'Rz')

glob_vs_glob(y_glob_v1, z_glob_v1, 'global_y', 'global_z', 'global_y_vs_global_z', legendlabels_v1)
