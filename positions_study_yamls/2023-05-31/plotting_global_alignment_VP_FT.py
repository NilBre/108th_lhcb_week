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
outname_prefix = 'outfiles_global_align/'



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
    if y_axis == 'Ty':
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
            plt.ylabel(f'mean {y_axis} module position [mm]')
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
    # new spacing
    # print('files:', files)
    path = 'retest_uncertainty/input_txt/loose_particles/global_alignment_files'
    if files[0] == f'{path}/v1/parsedlog_v1_global.json' or files[0] == f'{path}/v2/parsedlog_v2_global.json' or files[0] == f'{path}/v3/parsedlog_v3_global.json' or files[0] == f"{path}/retest_v1_to_v4/v1/parsedlog.json" or files[0] == "retest_uncertainty/json/parsedlog_500k_old_unc_loose.json" or files[0] == f'{path}/2023-11-07/v5/parsedlog.json' or files[0] == f'{path}/2023-11-07/v6/parsedlog.json' or files[0] == f'{path}/2023-11-07/v5_1/parsedlog.json' or files[0] == f'{path}/2023-11-07/v5_3/parsedlog.json' or files[0] == f'{path}/2023-11-10/v3_CFrames_TxRz/parsedlog.json' or files[0] == f'{path}/2023-11-09/v5_no_Cframes/parsedlog.json' or files[0] == f'{path}/2023-11-10/v4/parsedlog.json' or files[0] == f'{path}/2023-12-06/v10/parsedlog.json':
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
        # if withLongModules == True and withCFrames == False:
        #     print('withLongModules == True and withCFrames == False')
        #     runs_T1 = ['FT/T1/U/HL0/M0', 'FT/T1/U/HL0/M1', 'FT/T1/U/HL0/M2', 'FT/T1/U/HL0/M3', 'FT/T1/U/HL0/M4',
        #                'FT/T1/U/HL1/M0', 'FT/T1/U/HL1/M1', 'FT/T1/U/HL1/M2', 'FT/T1/U/HL1/M3', 'FT/T1/U/HL1/M4']
        #     runs_T2 = ['FT/T2/U/HL0/M0', 'FT/T2/U/HL0/M1', 'FT/T2/U/HL0/M2', 'FT/T2/U/HL0/M3', 'FT/T2/U/HL0/M4',
        #                'FT/T2/U/HL1/M0', 'FT/T2/U/HL1/M1', 'FT/T2/U/HL1/M2', 'FT/T2/U/HL1/M3', 'FT/T2/U/HL1/M4']
        #     runs = ['FT/T3/U/HL0/M0', 'FT/T3/U/HL0/M1', 'FT/T3/U/HL0/M2', 'FT/T3/U/HL0/M3', 'FT/T3/U/HL0/M4', # , 'FT/T3/U/HL0/M5'
        #             'FT/T3/U/HL1/M0', 'FT/T3/U/HL1/M1', 'FT/T3/U/HL1/M2', 'FT/T3/U/HL1/M3', 'FT/T3/U/HL1/M4'] # , 'FT/T3/U/HL0/M5'
        # if withLongModules == False and withCFrames == False:
        #     print('withLongModules == False and withCFrames == False')
        #     runs_T1 = ["FT/T1/U/HL0/Q0/M0", "FT/T1/U/HL0/Q0/M1", "FT/T1/U/HL0/Q0/M2", "FT/T1/U/HL0/Q0/M3", "FT/T1/U/HL0/Q0/M4",
        #                "FT/T1/U/HL0/Q2/M0", "FT/T1/U/HL0/Q2/M1", "FT/T1/U/HL0/Q2/M2", "FT/T1/U/HL0/Q2/M3", "FT/T1/U/HL0/Q2/M4",
        #                "FT/T1/U/HL1/Q1/M0", "FT/T1/U/HL1/Q1/M1", "FT/T1/U/HL1/Q1/M2", "FT/T1/U/HL1/Q1/M3", "FT/T1/U/HL1/Q1/M4",
        #                "FT/T1/U/HL1/Q3/M0", "FT/T1/U/HL1/Q3/M1", "FT/T1/U/HL1/Q3/M2", "FT/T1/U/HL1/Q3/M3", "FT/T1/U/HL1/Q3/M4"]
        #
        #     runs_T2 = ["FT/T2/U/HL0/Q0/M0", "FT/T2/U/HL0/Q0/M1", "FT/T2/U/HL0/Q0/M2", "FT/T2/U/HL0/Q0/M3", "FT/T2/U/HL0/Q0/M4",
        #                "FT/T2/U/HL0/Q2/M0", "FT/T2/U/HL0/Q2/M1", "FT/T2/U/HL0/Q2/M2", "FT/T2/U/HL0/Q2/M3", "FT/T2/U/HL0/Q2/M4",
        #                "FT/T2/U/HL1/Q1/M0", "FT/T2/U/HL1/Q1/M1", "FT/T2/U/HL1/Q1/M2", "FT/T2/U/HL1/Q1/M3", "FT/T2/U/HL1/Q1/M4",
        #                "FT/T2/U/HL1/Q3/M0", "FT/T2/U/HL1/Q3/M1", "FT/T2/U/HL1/Q3/M2", "FT/T2/U/HL1/Q3/M3", "FT/T2/U/HL1/Q3/M4"]
        #
        #     runs = ["FT/T3/U/HL0/Q0/M0", "FT/T3/U/HL0/Q0/M1", "FT/T3/U/HL0/Q0/M2", "FT/T3/U/HL0/Q0/M3", "FT/T3/U/HL0/Q0/M4",
        #             "FT/T3/U/HL0/Q2/M0", "FT/T3/U/HL0/Q2/M1", "FT/T3/U/HL0/Q2/M2", "FT/T3/U/HL0/Q2/M3", "FT/T3/U/HL0/Q2/M4",
        #             "FT/T3/U/HL1/Q1/M0", "FT/T3/U/HL1/Q1/M1", "FT/T3/U/HL1/Q1/M2", "FT/T3/U/HL1/Q1/M3", "FT/T3/U/HL1/Q1/M4",
        #             "FT/T3/U/HL1/Q3/M0", "FT/T3/U/HL1/Q3/M1", "FT/T3/U/HL1/Q3/M2", "FT/T3/U/HL1/Q3/M3", "FT/T3/U/HL1/Q3/M4"]
        # if withLongModules == False and withCFrames == True:
        #     print('withLongModules == False and withCFrames == True')
        #     runs_T1 = ['FT/T1/X1U/HL0', "FT/T1/U/HL0/Q0/M0", "FT/T1/U/HL0/Q0/M1", "FT/T1/U/HL0/Q0/M2", "FT/T1/U/HL0/Q0/M3", "FT/T1/U/HL0/Q0/M4",
        #                'FT/T1/X1U/HL1', "FT/T1/U/HL0/Q2/M0", "FT/T1/U/HL0/Q2/M1", "FT/T1/U/HL0/Q2/M2", "FT/T1/U/HL0/Q2/M3", "FT/T1/U/HL0/Q2/M4",
        #                'FT/T1/VX2/HL0', "FT/T1/U/HL1/Q1/M0", "FT/T1/U/HL1/Q1/M1", "FT/T1/U/HL1/Q1/M2", "FT/T1/U/HL1/Q1/M3", "FT/T1/U/HL1/Q1/M4",
        #                'FT/T1/VX2/HL1', "FT/T1/U/HL1/Q3/M0", "FT/T1/U/HL1/Q3/M1", "FT/T1/U/HL1/Q3/M2", "FT/T1/U/HL1/Q3/M3", "FT/T1/U/HL1/Q3/M4"]
        #
        #     runs_T2 = ['FT/T2/X1U/HL0', "FT/T2/U/HL0/Q0/M0", "FT/T2/U/HL0/Q0/M1", "FT/T2/U/HL0/Q0/M2", "FT/T2/U/HL0/Q0/M3", "FT/T2/U/HL0/Q0/M4",
        #                'FT/T2/X1U/HL1', "FT/T2/U/HL0/Q2/M0", "FT/T2/U/HL0/Q2/M1", "FT/T2/U/HL0/Q2/M2", "FT/T2/U/HL0/Q2/M3", "FT/T2/U/HL0/Q2/M4",
        #                'FT/T2/VX2/HL0', "FT/T2/U/HL1/Q1/M0", "FT/T2/U/HL1/Q1/M1", "FT/T2/U/HL1/Q1/M2", "FT/T2/U/HL1/Q1/M3", "FT/T2/U/HL1/Q1/M4",
        #                'FT/T2/VX2/HL1', "FT/T2/U/HL1/Q3/M0", "FT/T2/U/HL1/Q3/M1", "FT/T2/U/HL1/Q3/M2", "FT/T2/U/HL1/Q3/M3", "FT/T2/U/HL1/Q3/M4"]
        #
        #     runs = ['FT/T3/X1U/HL0', "FT/T3/U/HL0/Q0/M0", "FT/T3/U/HL0/Q0/M1", "FT/T3/U/HL0/Q0/M2", "FT/T3/U/HL0/Q0/M3", "FT/T3/U/HL0/Q0/M4",
        #             'FT/T3/X1U/HL1', "FT/T3/U/HL0/Q2/M0", "FT/T3/U/HL0/Q2/M1", "FT/T3/U/HL0/Q2/M2", "FT/T3/U/HL0/Q2/M3", "FT/T3/U/HL0/Q2/M4",
        #             'FT/T3/VX2/HL0', "FT/T3/U/HL1/Q1/M0", "FT/T3/U/HL1/Q1/M1", "FT/T3/U/HL1/Q1/M2", "FT/T3/U/HL1/Q1/M3", "FT/T3/U/HL1/Q1/M4",
        #             'FT/T3/VX2/HL1', "FT/T3/U/HL1/Q3/M0", "FT/T3/U/HL1/Q3/M1", "FT/T3/U/HL1/Q3/M2", "FT/T3/U/HL1/Q3/M3", "FT/T3/U/HL1/Q3/M4"]
        # if withLongModules == True and withCFrames == True:
        #     print('withLongModules == True and withCFrames == True')
        #     runs_T1 = ['FT/T1/X1U/HL0', 'FT/T1/X1U/HL1', 'FT/T1/VX2/HL0', 'FT/T1/VX2/HL1',
        #                'FT/T1/U/HL0/M0', 'FT/T1/U/HL0/M1', 'FT/T1/U/HL0/M2', 'FT/T1/U/HL0/M3', 'FT/T1/U/HL0/M4',
        #                'FT/T1/U/HL1/M0', 'FT/T1/U/HL1/M1', 'FT/T1/U/HL1/M2', 'FT/T1/U/HL1/M3', 'FT/T1/U/HL1/M4',
        #                "FT/T1/U/HL0/Q0/M0", "FT/T1/U/HL0/Q0/M1", "FT/T1/U/HL0/Q0/M2", "FT/T1/U/HL0/Q0/M3", "FT/T1/U/HL0/Q0/M4",
        #                "FT/T1/U/HL0/Q2/M0", "FT/T1/U/HL0/Q2/M1", "FT/T1/U/HL0/Q2/M2", "FT/T1/U/HL0/Q2/M3", "FT/T1/U/HL0/Q2/M4",
        #                "FT/T1/U/HL1/Q1/M0", "FT/T1/U/HL1/Q1/M1", "FT/T1/U/HL1/Q1/M2", "FT/T1/U/HL1/Q1/M3", "FT/T1/U/HL1/Q1/M4",
        #                "FT/T1/U/HL1/Q3/M0", "FT/T1/U/HL1/Q3/M1", "FT/T1/U/HL1/Q3/M2", "FT/T1/U/HL1/Q3/M3", "FT/T1/U/HL1/Q3/M4"]
        #
        #     runs_T2 = ['FT/T2/X1U/HL0', 'FT/T2/X1U/HL1', 'FT/T2/VX2/HL0', 'FT/T2/VX2/HL1',
        #                'FT/T2/U/HL0/M0', 'FT/T2/U/HL0/M1', 'FT/T2/U/HL0/M2', 'FT/T2/U/HL0/M3', 'FT/T2/U/HL0/M4',
        #                'FT/T2/U/HL1/M0', 'FT/T2/U/HL1/M1', 'FT/T2/U/HL1/M2', 'FT/T2/U/HL1/M3', 'FT/T2/U/HL1/M4',
        #                "FT/T2/U/HL0/Q0/M0", "FT/T2/U/HL0/Q0/M1", "FT/T2/U/HL0/Q0/M2", "FT/T2/U/HL0/Q0/M3", "FT/T2/U/HL0/Q0/M4",
        #                "FT/T2/U/HL0/Q2/M0", "FT/T2/U/HL0/Q2/M1", "FT/T2/U/HL0/Q2/M2", "FT/T2/U/HL0/Q2/M3", "FT/T2/U/HL0/Q2/M4",
        #                "FT/T2/U/HL1/Q1/M0", "FT/T2/U/HL1/Q1/M1", "FT/T2/U/HL1/Q1/M2", "FT/T2/U/HL1/Q1/M3", "FT/T2/U/HL1/Q1/M4",
        #                "FT/T2/U/HL1/Q3/M0", "FT/T2/U/HL1/Q3/M1", "FT/T2/U/HL1/Q3/M2", "FT/T2/U/HL1/Q3/M3", "FT/T2/U/HL1/Q3/M4"]
        #
        #     runs = ['FT/T3/X1U/HL0', 'FT/T3/X1U/HL1', 'FT/T3/VX2/HL0', 'FT/T3/VX2/HL1',
        #             'FT/T3/U/HL0/M0', 'FT/T3/U/HL0/M1', 'FT/T3/U/HL0/M2', 'FT/T3/U/HL0/M3', 'FT/T3/U/HL0/M4',
        #             'FT/T3/U/HL1/M0', 'FT/T3/U/HL1/M1', 'FT/T3/U/HL1/M2', 'FT/T3/U/HL1/M3', 'FT/T3/U/HL1/M4',
        #             "FT/T3/U/HL0/Q0/M0", "FT/T3/U/HL0/Q0/M1", "FT/T3/U/HL0/Q0/M2", "FT/T3/U/HL0/Q0/M3", "FT/T3/U/HL0/Q0/M4",
        #             "FT/T3/U/HL0/Q2/M0", "FT/T3/U/HL0/Q2/M1", "FT/T3/U/HL0/Q2/M2", "FT/T3/U/HL0/Q2/M3", "FT/T3/U/HL0/Q2/M4",
        #             "FT/T3/U/HL1/Q1/M0", "FT/T3/U/HL1/Q1/M1", "FT/T3/U/HL1/Q1/M2", "FT/T3/U/HL1/Q1/M3", "FT/T3/U/HL1/Q1/M4",
        #             "FT/T3/U/HL1/Q3/M0", "FT/T3/U/HL1/Q3/M1", "FT/T3/U/HL1/Q3/M2", "FT/T3/U/HL1/Q3/M3", "FT/T3/U/HL1/Q3/M4"]
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

path = 'retest_uncertainty/input_txt/loose_particles/global_alignment_files'
path2 = 'retest_uncertainty/input_txt/loose_particles'
'''
    configureGlobalAlignment(halfdofs="TxTyTzRy") from AlignmentScenarios.py in Humboldt (Alignment)
'''
files_v1 = [\
        f"{path}/v1/parsedlog_v1_global.json",
        # f"{path}/v1_1/parsedlog_v1_1_global.json",
        f"{path}/v1_2/parsedlog_v1_2_smallRxUnc_10mu.json",
        "retest_uncertainty/json/parsedlog_TxRxRz_smallRxSurveyUnc.json",
        # f'{path}/v4_align_scenarios/parsedlog.json',
        # f'{path}/vp_only/parsedlog_vp_only.json',
        f'{path2}/v4_with_and_no_backframes/v4_no_backframes/parsedlog_no_backframes_v4.json',
        f'{path2}/v4_with_and_no_backframes/v4_with_backframes/parsedlog_with_backframes_v4.json',
        f"{path}/2023-10-31/T1_added/parsedlog_T1.json",
        f"{path}/2023-10-31/T2_added/parsedlog_T2.json",
]
legendlabels_v1=[\
              "global v1, wouter joints",
              # "global v1, 10mu Tx",
              "global v1, 10mu Tx, smallRxUnc",
              "TxRxRz, smallSurveyUnc",
              # 'v4',
              # 'velo_only',
              'v4_no_backframes',
              'v4_with_backframes',
              'T1_added',
              'T2_added',
]

'''
    configureGlobalAlignment_v2(halfdofs="TxTyTzRy") from AlignmentScenarios.py in Humboldt (Alignment)
'''
files_v2 = [\
        f"{path}/v2/parsedlog_v2_global.json",
        f"{path}/v2_1/parsedlog_v2_1_global.json",
        f"{path}/v2_2/parsedlog_v2_2_smallRxUnc_10mu.json",
        "retest_uncertainty/json/parsedlog_TxRxRz_smallRxSurveyUnc.json",
]
legendlabels_v2=[\
              "global v2, wouter joints",
              "global v2, 10mu Tx",
              "global v2, 10mu Tx, smallRxUnc",
              "TxRxRz, smallSurveyUnc"
]

'''
    configureGlobalAlignment_v3(halfdofs="TxTyTzRy") from AlignmentScenarios.py in Humboldt (Alignment)
'''
files_v3 = [\
        f"{path}/v3/parsedlog_v3_global.json",
        f"{path}/v3_1/parsedlog_v3_1_global.json",
        f"{path}/v3_2/parsedlog_v3_2_smallRxUnc_10mu.json",
        "retest_uncertainty/json/parsedlog_TxRxRz_smallRxSurveyUnc.json",
]
legendlabels_v3=[\
              "global v3, wouter joints",
              "global v3, 10mu Tx",
              "global v3, 10mu Tx, smallRxUnc",
              "TxRxRz, smallSurveyUnc"
]

# v1 to v4
f_retest = [\
        f"{path}/retest_v1_to_v4/v1/parsedlog.json",
        # f"{path}/v1_2/parsedlog_v1_2_smallRxUnc_10mu.json",
        # f"{path}/retest_v1_to_v4/v2/parsedlog.json",
        # f"{path}/retest_v1_to_v4/v3/parsedlog.json",
        f"{path}/retest_v1_to_v4/v4/parsedlog.json",
        f'{path}/2023-11-07/v1_with_velo/parsedlog.json',
        f'{path}/2023-11-07/v4_with_velo/parsedlog.json',
        "retest_uncertainty/json/parsedlog_TxRxRz_smallRxSurveyUnc.json",
]
legendlabels_retest=[\
        'v1',
        # 'v1, smallSurveyRxUnc',
        # 'v2',
        # 'v3',
        'v4',
        'v1_with_velo',
        'v4_with_velo',
        'SciFi, TxRxRz, small Rx surveyUnc'
]

# 2023-11-07
v5_var = [\
        f'{path}/2023-11-07/v5/parsedlog.json',
        f'{path}/2023-11-07/v5_1/parsedlog.json',
        f'{path}/2023-11-07/v5_2/parsedlog.json',
]

v5_labels = [\
    'v5',
    'v5_1',
    'v5_2'
]

v6_var = [\
        f'{path}/2023-11-07/v6/parsedlog.json',
        f'{path}/2023-11-07/v6_1/parsedlog.json',
        f'{path}/2023-11-07/v6_2/parsedlog.json',
]

v6_labels = [\
    'v6',
    'v6_1',
    'v6_2'
]

v56_comp = [\
        f'{path}/2023-11-07/v5_1/parsedlog.json',
        f'{path}/2023-11-07/v5_2/parsedlog.json',
        f'{path}/2023-11-07/v6_2/parsedlog.json',
        # f"{path}/retest_v1_to_v4/v1/parsedlog.json",
]

v56_comp_labels = [\
    'v5_1',
    'v5_2',
    'v6_2',
    # 'v1',
]

mod_and_halfmod = [\
    f'{path}/2023-11-07/v5_3/parsedlog.json',
    f'{path}/2023-11-07/v6_3/parsedlog.json',
]

mod_and_halfmod_labels = [\
    'modules and halfmodules TxRz', # v5_3
    'modules and halfmodules TxRxRz', # v6_3
]

cframes_studies = [\
    f'{path}/2023-11-09/v5_no_Cframes/parsedlog.json',
    f'{path}/2023-11-09/v1_CFrames_Tz/parsedlog.json',
    f'{path}/2023-11-09/v2_CFrames_Rz/parsedlog.json',
    f'{path}/2023-12-01/v5_3_nominal_pos/parsedlog.json',
    # f'{path}/2023-11-09/v6_CFrames_Tx/parsedlog.json',
    # f'{path}/2023-11-09/v3_CFrames_TxRz/parsedlog.json',
]

cframes_labels = [\
    'v5_no_CFrames',
    'v1_CFrames_Tz',
    'v2_CFrames_Rz',
    'v5_3_nominal',
    # 'v6_CFrames_Tx',
    # 'v3_CFrames_TxRz',
]

velo_long_tracks = [\
    f'{path}/2023-11-10/v4/parsedlog.json',
    f'{path}/2023-11-10/v5/parsedlog.json',
    f'{path}/2023-11-10/v6/parsedlog.json',
]

velo_long_tracks_labels = [\
    'v4',
    'v5',
    'v6'
]

full_v1_tests = [\
    f'{path}/v1/parsedlog_v1_global.json',
    f'{path}/2023-11-07/v1_veloRx/parsedlog.json',
    f'{path}/2023-11-07/v1_veloRz/parsedlog.json',
    f'{path}/2023-11-07/v1_with_velo/parsedlog.json',
    f'{path}/2023-11-07/v5_3/parsedlog.json',
    f'{path}/2023-11-07/v6_3/parsedlog.json',
]

full_v1_labels = [\
    'v1',
    'v1 + Velo Rx, smallRxUnc',
    'v1 + Velo Rz, smallRxUnc',
    'v1 + Velo RxRz, smallRxUnc',
    'modules and halfmodules TxRz',
    'modules and halfmodules TxRxRz',
]

# config: modules, halfmodules, cframes, VP global with CFrame survey and small uncertainties in Rx
new_conf = [\
    f'{path}/2023-11-10/v3_CFrames_TxRz/parsedlog.json',
    f'{path}/2023-11-10/v6_CFrames_Tx/parsedlog.json'
]

new_conf_labels = [\
    'v3_CFrames_TxRz',
    'v6_CFrames_Tx',
]
# date: 203-12-06: v10 and first log modules then half modules
combi = [\
    f'{path}/2023-12-06/v10/parsedlog.json',
    # f'{path}/2023-12-06/first_half_then_long_modules/parsedlog_iter5_after_longmodules.json',
    f'{path}/2023-12-06/first_half_then_long_modules/parsedlog_iter14_after_halfmodules.json'
]

combi_labels = [\
    'CFr. TxTz, HMod. TxRxRz', #
    # 'long+half mod, iter 5',
    'long+half mod, iter 14',
]

survey_module_positions = 'survey/survey_Modules.yml'

layers = ['T1U', 'T1V', 'T1X1', 'T1X2', 'T2U', 'T2V', 'T2X1', 'T2X2', 'T3U', 'T3V', 'T3X1', 'T3X2']

runs = ["T3UHL0Q0M0", "T3UHL0Q0M1", "T3UHL0Q0M2", "T3UHL0Q0M3", "T3UHL0Q0M4", "T3UHL0Q2M0", "T3UHL0Q2M1", "T3UHL0Q2M2", "T3UHL0Q2M3", "T3UHL0Q2M4", "T3UHL1Q1M0"#\
        , "T3UHL1Q1M1", "T3UHL1Q1M2", "T3UHL1Q1M3", "T3UHL1Q1M4", "T3UHL1Q3M0", "T3UHL1Q3M1", "T3UHL1Q3M2", "T3UHL1Q3M3", "T3UHL1Q3M4"]

plotting_variables = ['Tx', 'Ty', 'Tz', 'nHits', 'nTracks', 'x_global', 'y_global', 'z_global']
red_blue_vars = ['Tx', 'Ty', 'Tz', 'Rx', 'Ry', 'Rz', 'nHits', 'nTracks', 'x_global', 'y_global', 'z_global']
print('v1')
align_outputs_v1 = make_outfiles(files_v1, red_blue_vars) # , withLongModules=False, withCFrames=False
print('v2')
align_outputs_v2 = make_outfiles(files_v2, red_blue_vars) # , withLongModules=False, withCFrames=False
print('v3')
align_outputs_v3 = make_outfiles(files_v3, red_blue_vars) # , withLongModules=False, withCFrames=False
print('output retest')
align_outputs_retest = make_outfiles(f_retest, red_blue_vars) # , withLongModules=False, withCFrames=False
print('v5')
align_out_v5 = make_outfiles(v5_var, red_blue_vars) # , withLongModules=True, withCFrames=True
print('v6')
align_out_v6 = make_outfiles(v6_var, red_blue_vars) # , withLongModules=True, withCFrames=True
print('v56 comp')
align_out_v56_comp = make_outfiles(v56_comp, red_blue_vars) # , withLongModules=True, withCFrames=True
print('modules and halfmodules')
align_out_mod_and_halfmod = make_outfiles(mod_and_halfmod, red_blue_vars) # , withLongModules=True, withCFrames=False

align_out_full_v1 = make_outfiles(full_v1_tests, red_blue_vars) # , withLongModules=True, withCFrames=False
align_out_new_conf = make_outfiles(new_conf, red_blue_vars) # , withLongModules=True, withCFrames=True
align_out_cframes_studies = make_outfiles(cframes_studies, red_blue_vars) # , withLongModules=True, withCFrames=True
align_out_velo_long_tracks = make_outfiles(velo_long_tracks, red_blue_vars) # , withLongModules=True, withCFrames=False
align_out_combi = make_outfiles(combi, red_blue_vars) # , withLongModules=True, withCFrames=True

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

Tx_retest      = align_outputs_retest[0]
Ty_retest      = align_outputs_retest[1]
Tz_retest      = align_outputs_retest[2]
Rx_retest      = align_outputs_retest[3]
Ry_retest      = align_outputs_retest[4]
Rz_retest      = align_outputs_retest[5]
nHits_retest   = align_outputs_retest[6]
nTracks_retest = align_outputs_retest[7]
x_glob_retest  = align_outputs_retest[8]
y_glob_retest  = align_outputs_retest[9]
z_glob_retest  = align_outputs_retest[10]

Tx_v5      = align_out_v5[0]
Ty_v5      = align_out_v5[1]
Tz_v5      = align_out_v5[2]
Rx_v5      = align_out_v5[3]
Ry_v5      = align_out_v5[4]
Rz_v5      = align_out_v5[5]
nHits_v5   = align_out_v5[6]
nTracks_v5 = align_out_v5[7]
x_glob_v5  = align_out_v5[8]
y_glob_v5  = align_out_v5[9]
z_glob_v5  = align_out_v5[10]

Tx_v6      = align_out_v6[0]
Ty_v6      = align_out_v6[1]
Tz_v6      = align_out_v6[2]
Rx_v6      = align_out_v6[3]
Ry_v6      = align_out_v6[4]
Rz_v6      = align_out_v6[5]
nHits_v6   = align_out_v6[6]
nTracks_v6 = align_out_v6[7]
x_glob_v6  = align_out_v6[8]
y_glob_v6  = align_out_v6[9]
z_glob_v6  = align_out_v6[10]

Tx_v56      = align_out_v56_comp[0]
Ty_v56      = align_out_v56_comp[1]
Tz_v56      = align_out_v56_comp[2]
Rx_v56      = align_out_v56_comp[3]
Ry_v56      = align_out_v56_comp[4]
Rz_v56      = align_out_v56_comp[5]
nHits_v56   = align_out_v56_comp[6]
nTracks_v56 = align_out_v56_comp[7]
x_glob_v56  = align_out_v56_comp[8]
y_glob_v56  = align_out_v56_comp[9]
z_glob_v56  = align_out_v56_comp[10]

Tx_mod_and_halfmod      = align_out_mod_and_halfmod[0]
Ty_mod_and_halfmod      = align_out_mod_and_halfmod[1]
Tz_mod_and_halfmod      = align_out_mod_and_halfmod[2]
Rx_mod_and_halfmod      = align_out_mod_and_halfmod[3]
Ry_mod_and_halfmod      = align_out_mod_and_halfmod[4]
Rz_mod_and_halfmod      = align_out_mod_and_halfmod[5]
nHits_mod_and_halfmod   = align_out_mod_and_halfmod[6]
nTracks_mod_and_halfmod = align_out_mod_and_halfmod[7]
x_glob_mod_and_halfmod  = align_out_mod_and_halfmod[8]
y_glob_mod_and_halfmod  = align_out_mod_and_halfmod[9]
z_glob_mod_and_halfmod  = align_out_mod_and_halfmod[10]

Tx_full_v1      = align_out_full_v1[0]
Ty_full_v1      = align_out_full_v1[1]
Tz_full_v1      = align_out_full_v1[2]
Rx_full_v1      = align_out_full_v1[3]
Ry_full_v1      = align_out_full_v1[4]
Rz_full_v1      = align_out_full_v1[5]
nHits_full_v1   = align_out_full_v1[6]
nTracks_full_v1 = align_out_full_v1[7]
x_glob_full_v1  = align_out_full_v1[8]
y_glob_full_v1  = align_out_full_v1[9]
z_glob_full_v1  = align_out_full_v1[10]

Tx_new_conf      = align_out_new_conf[0]
Ty_new_conf      = align_out_new_conf[1]
Tz_new_conf      = align_out_new_conf[2]
Rx_new_conf      = align_out_new_conf[3]
Ry_new_conf      = align_out_new_conf[4]
Rz_new_conf      = align_out_new_conf[5]
nHits_new_conf   = align_out_new_conf[6]
nTracks_new_conf = align_out_new_conf[7]
x_glob_new_conf  = align_out_new_conf[8]
y_glob_new_conf  = align_out_new_conf[9]
z_glob_new_conf  = align_out_new_conf[10]

# cframes studies for global alignment, 2023-11-28
Tx_cframes_studies      = align_out_cframes_studies[0]
Ty_cframes_studies      = align_out_cframes_studies[1]
Tz_cframes_studies      = align_out_cframes_studies[2]
Rx_cframes_studies      = align_out_cframes_studies[3]
Ry_cframes_studies      = align_out_cframes_studies[4]
Rz_cframes_studies      = align_out_cframes_studies[5]
nHits_cframes_studies   = align_out_cframes_studies[6]
nTracks_cframes_studies = align_out_cframes_studies[7]
x_glob_cframes_studies  = align_out_cframes_studies[8]
y_glob_cframes_studies  = align_out_cframes_studies[9]
z_glob_cframes_studies  = align_out_cframes_studies[10]

# backwards tracks + velo tracks, 2023-11-28
Tx_velo_long_tracks      = align_out_velo_long_tracks[0]
Ty_velo_long_tracks      = align_out_velo_long_tracks[1]
Tz_velo_long_tracks      = align_out_velo_long_tracks[2]
Rx_velo_long_tracks      = align_out_velo_long_tracks[3]
Ry_velo_long_tracks      = align_out_velo_long_tracks[4]
Rz_velo_long_tracks      = align_out_velo_long_tracks[5]
nHits_velo_long_tracks   = align_out_velo_long_tracks[6]
nTracks_velo_long_tracks = align_out_velo_long_tracks[7]
x_glob_velo_long_tracks  = align_out_velo_long_tracks[8]
y_glob_velo_long_tracks  = align_out_velo_long_tracks[9]
z_glob_velo_long_tracks  = align_out_velo_long_tracks[10]

Tx_combi      = align_out_combi[0]
Ty_combi      = align_out_combi[1]
Tz_combi      = align_out_combi[2]
Rx_combi      = align_out_combi[3]
Ry_combi      = align_out_combi[4]
Rz_combi      = align_out_combi[5]
nHits_combi   = align_out_combi[6]
nTracks_combi = align_out_combi[7]
x_glob_combi  = align_out_combi[8]
y_glob_combi  = align_out_combi[9]
z_glob_combi  = align_out_combi[10]

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

    Tx_data_retest = Tx_retest[n]
    Ty_data_retest = Ty_retest[n]
    Tz_data_retest = Tz_retest[n]
    Rx_data_retest = Rx_retest[n]
    Ry_data_retest = Ry_retest[n]
    Rz_data_retest = Rz_retest[n]
    x_g_retest = x_glob_retest[n]
    y_g_retest = y_glob_retest[n]
    z_g_retest = z_glob_retest[n]

    Tx_data_v5 = Tx_v5[n]
    Ty_data_v5 = Ty_v5[n]
    Tz_data_v5 = Tz_v5[n]
    Rx_data_v5 = Rx_v5[n]
    Ry_data_v5 = Ry_v5[n]
    Rz_data_v5 = Rz_v5[n]
    x_g_v5 = x_glob_v5[n]
    y_g_v5 = y_glob_v5[n]
    z_g_v5 = z_glob_v5[n]

    Tx_data_v6 = Tx_v6[n]
    Ty_data_v6 = Ty_v6[n]
    Tz_data_v6 = Tz_v6[n]
    Rx_data_v6 = Rx_v6[n]
    Ry_data_v6 = Ry_v6[n]
    Rz_data_v6 = Rz_v6[n]
    x_g_v6 = x_glob_v6[n]
    y_g_v6 = y_glob_v6[n]
    z_g_v6 = z_glob_v6[n]

    Tx_data_v56 = Tx_v56[n]
    Ty_data_v56 = Ty_v56[n]
    Tz_data_v56 = Tz_v56[n]
    Rx_data_v56 = Rx_v56[n]
    Ry_data_v56 = Ry_v56[n]
    Rz_data_v56 = Rz_v56[n]
    x_g_v56 = x_glob_v56[n]
    y_g_v56 = y_glob_v56[n]
    z_g_v56 = z_glob_v56[n]

    Tx_data_mod_and_halfmod = Tx_mod_and_halfmod[n]
    Ty_data_mod_and_halfmod = Ty_mod_and_halfmod[n]
    Tz_data_mod_and_halfmod = Tz_mod_and_halfmod[n]
    Rx_data_mod_and_halfmod = Rx_mod_and_halfmod[n]
    Ry_data_mod_and_halfmod = Ry_mod_and_halfmod[n]
    Rz_data_mod_and_halfmod = Rz_mod_and_halfmod[n]
    x_g_mod_and_halfmod = x_glob_mod_and_halfmod[n]
    y_g_mod_and_halfmod = y_glob_mod_and_halfmod[n]
    z_g_mod_and_halfmod = z_glob_mod_and_halfmod[n]

    Tx_data_full_v1 = Tx_full_v1[n]
    Ty_data_full_v1 = Ty_full_v1[n]
    Tz_data_full_v1 = Tz_full_v1[n]
    Rx_data_full_v1 = Rx_full_v1[n]
    Ry_data_full_v1 = Ry_full_v1[n]
    Rz_data_full_v1 = Rz_full_v1[n]
    x_g_full_v1 = x_glob_full_v1[n]
    y_g_full_v1 = y_glob_full_v1[n]
    z_g_full_v1 = z_glob_full_v1[n]

    Tx_data_new_conf = Tx_new_conf[n]
    Ty_data_new_conf = Ty_new_conf[n]
    Tz_data_new_conf = Tz_new_conf[n]
    Rx_data_new_conf = Rx_new_conf[n]
    Ry_data_new_conf = Ry_new_conf[n]
    Rz_data_new_conf = Rz_new_conf[n]
    x_g_new_conf = x_glob_new_conf[n]
    y_g_new_conf = y_glob_new_conf[n]
    z_g_new_conf = z_glob_new_conf[n]

    # cframes
    Tx_data_cframes_studies = Tx_cframes_studies[n]
    Ty_data_cframes_studies = Ty_cframes_studies[n]
    Tz_data_cframes_studies = Tz_cframes_studies[n]
    Rx_data_cframes_studies = Rx_cframes_studies[n]
    Ry_data_cframes_studies = Ry_cframes_studies[n]
    Rz_data_cframes_studies = Rz_cframes_studies[n]
    x_g_cframes_studies = x_glob_cframes_studies[n]
    y_g_cframes_studies = y_glob_cframes_studies[n]
    z_g_cframes_studies = z_glob_cframes_studies[n]

    # backwards tracks + long tracks
    Tx_data_velo_long_tracks = Tx_velo_long_tracks[n]
    Ty_data_velo_long_tracks = Ty_velo_long_tracks[n]
    Tz_data_velo_long_tracks = Tz_velo_long_tracks[n]
    Rx_data_velo_long_tracks = Rx_velo_long_tracks[n]
    Ry_data_velo_long_tracks = Ry_velo_long_tracks[n]
    Rz_data_velo_long_tracks = Rz_velo_long_tracks[n]
    x_g_velo_long_tracks = x_glob_velo_long_tracks[n]
    y_g_velo_long_tracks = y_glob_velo_long_tracks[n]
    z_g_velo_long_tracks = z_glob_velo_long_tracks[n]

    Tx_data_combi = Tx_combi[n]
    Ty_data_combi = Ty_combi[n]
    Tz_data_combi = Tz_combi[n]
    Rx_data_combi = Rx_combi[n]
    Ry_data_combi = Ry_combi[n]
    Rz_data_combi = Rz_combi[n]
    x_g_combi = x_glob_combi[n]
    y_g_combi = y_glob_combi[n]
    z_g_combi = z_glob_combi[n]

    plot_x_y_constants(x_g_v1, y_g_v1, Tx_data_v1, Ty_data_v1, legendlabels_v1, layers[n], 'quarter', 'global', 'v1_x_vs_z')
    plot_x_y_constants(x_g_v2, y_g_v2, Tx_data_v2, Ty_data_v2, legendlabels_v2, layers[n], 'quarter', 'global', 'v2_x_vs_z')
    plot_x_y_constants(x_g_v3, y_g_v3, Tx_data_v3, Ty_data_v3, legendlabels_v3, layers[n], 'quarter', 'global', 'v3_x_vs_z')
    plot_x_y_constants(x_g_retest, y_g_retest, Tx_data_retest, Ty_data_retest, legendlabels_retest, layers[n], 'quarter', 'global', 'retest_x_vs_z')
    plot_x_y_constants(x_g_v5, y_g_v5, Tx_data_v5, Ty_data_v5, v5_labels, layers[n], 'quarter', 'global', 'v5_x_vs_z')
    plot_x_y_constants(x_g_v6, y_g_v6, Tx_data_v6, Ty_data_v6, v6_labels, layers[n], 'quarter', 'global', 'v6_x_vs_z')
    plot_x_y_constants(x_g_v56, y_g_v56, Tx_data_v56, Ty_data_v56, v56_comp_labels, layers[n], 'quarter', 'global', 'v56_comp_x_vs_z')
    plot_x_y_constants(x_g_mod_and_halfmod, y_g_mod_and_halfmod, Tx_data_mod_and_halfmod, Ty_data_mod_and_halfmod, mod_and_halfmod_labels, layers[n], 'quarter', 'global', 'mod_halfmod_x_vs_z')
    plot_x_y_constants(x_g_full_v1, y_g_full_v1, Tx_data_full_v1, Ty_data_full_v1, full_v1_labels, layers[n], 'quarter', 'global', 'full_v1_x_vs_z')
    plot_x_y_constants(x_g_new_conf, y_g_new_conf, Tx_data_new_conf, Ty_data_new_conf, new_conf_labels, layers[n], 'quarter', 'global', 'new_conf_x_vs_z')
    plot_x_y_constants(x_g_cframes_studies, y_g_cframes_studies, Tx_data_cframes_studies, Ty_data_cframes_studies, cframes_labels, layers[n], 'quarter', 'global', 'cframes_studies_x_vs_z')
    plot_x_y_constants(x_g_velo_long_tracks, y_g_velo_long_tracks, Tx_data_velo_long_tracks, Ty_data_velo_long_tracks, velo_long_tracks_labels, layers[n], 'quarter', 'global', 'velo_long_tracks_x_vs_z')
    plot_x_y_constants(x_g_combi, y_g_combi, Tx_data_combi, Ty_data_combi, combi_labels, layers[n], 'quarter', 'global', 'combi_x_vs_z')

    plot_x_y_constants(x_g_v1, y_g_v1, Tx_data_v1, Ty_data_v1, legendlabels_v1, layers[n], 'quarter', 'local', 'v1_x_vs_y_local')
    plot_x_y_constants(x_g_v2, y_g_v2, Tx_data_v2, Ty_data_v2, legendlabels_v2, layers[n], 'quarter', 'local', 'v2_x_vs_y_local')
    plot_x_y_constants(x_g_v3, y_g_v3, Tx_data_v3, Ty_data_v3, legendlabels_v3, layers[n], 'quarter', 'local', 'v3_x_vs_y_local')
    plot_x_y_constants(x_g_retest, y_g_retest, Tx_data_retest, Ty_data_retest, legendlabels_retest, layers[n], 'quarter', 'local', 'retest_x_vs_y_local')
    plot_x_y_constants(x_g_v5, y_g_v5, Tx_data_v5, Ty_data_v5, v5_labels, layers[n], 'quarter', 'local', 'v5_x_vs_z_local')
    plot_x_y_constants(x_g_v6, y_g_v6, Tx_data_v6, Ty_data_v6, v6_labels, layers[n], 'quarter', 'local', 'v6_x_vs_z_local')
    plot_x_y_constants(x_g_v56, y_g_v56, Tx_data_v56, Ty_data_v56, v56_comp_labels, layers[n], 'quarter', 'local', 'v56_comp_x_vs_z_local')
    plot_x_y_constants(x_g_mod_and_halfmod, y_g_mod_and_halfmod, Tx_data_mod_and_halfmod, Ty_data_mod_and_halfmod, mod_and_halfmod_labels, layers[n], 'quarter', 'local', 'mod_halfmod_x_vs_z_local')
    plot_x_y_constants(x_g_full_v1, y_g_full_v1, Tx_data_full_v1, Ty_data_full_v1, full_v1_labels, layers[n], 'quarter', 'local', 'full_v1_x_vs_z_local')
    plot_x_y_constants(x_g_new_conf, y_g_new_conf, Tx_data_new_conf, Ty_data_new_conf, new_conf_labels, layers[n], 'quarter', 'local', 'new_conf_x_vs_z_local')
    plot_x_y_constants(x_g_cframes_studies, y_g_cframes_studies, Tx_data_cframes_studies, Ty_data_cframes_studies, cframes_labels, layers[n], 'quarter', 'local', 'cframes_studies_x_vs_z_local')
    plot_x_y_constants(x_g_velo_long_tracks, y_g_velo_long_tracks, Tx_data_velo_long_tracks, Ty_data_velo_long_tracks, velo_long_tracks_labels, layers[n], 'quarter', 'local', 'velo_long_tracks_x_vs_z_local')
    plot_x_y_constants(x_g_combi, y_g_combi, Tx_data_combi, Ty_data_combi, combi_labels, layers[n], 'quarter', 'local', 'combi_x_vs_z_local')

    check_module_edges(x_g_v1, y_g_v1, Tx_data_v1, Ty_data_v1, legendlabels_v1, layers[n], 'layer', 'global', Rx_data_v1, 'v1_global_align')
    check_module_edges(x_g_v2, y_g_v2, Tx_data_v2, Ty_data_v2, legendlabels_v2, layers[n], 'layer', 'global', Rx_data_v2, 'v2_global_align')
    check_module_edges(x_g_v3, y_g_v3, Tx_data_v3, Ty_data_v3, legendlabels_v3, layers[n], 'layer', 'global', Rx_data_v3, 'v3_global_align')
    check_module_edges(x_g_retest, y_g_retest, Tx_data_retest, Ty_data_retest, legendlabels_retest, layers[n], 'layer', 'global', Rx_data_retest, 'retest_global_align')
    check_module_edges(x_g_v5, y_g_v5, Tx_data_v5, Ty_data_v5, v5_labels, layers[n], 'layer', 'global', Rx_data_v5, 'v5_global_align')
    check_module_edges(x_g_v6, y_g_v6, Tx_data_v6, Ty_data_v6, v6_labels, layers[n], 'layer', 'global', Rx_data_v6, 'v6_global_align')
    check_module_edges(x_g_v56, y_g_v56, Tx_data_v56, Ty_data_v56, v56_comp_labels, layers[n], 'layer', 'global', Rx_data_v56, 'v56_comp_global_align')
    check_module_edges(x_g_mod_and_halfmod, y_g_mod_and_halfmod, Tx_data_mod_and_halfmod, Ty_data_mod_and_halfmod, mod_and_halfmod_labels, layers[n], 'layer', 'global', Rx_data_mod_and_halfmod, 'mod_halfmod_global_align')
    check_module_edges(x_g_full_v1, y_g_full_v1, Tx_data_full_v1, Ty_data_full_v1, full_v1_labels, layers[n], 'layer', 'global', Rx_data_full_v1, 'full_v1_global_align')
    check_module_edges(x_g_new_conf, y_g_new_conf, Tx_data_new_conf, Ty_data_new_conf, new_conf_labels, layers[n], 'layer', 'global', Rx_data_new_conf, 'new_conf_global_align')
    check_module_edges(x_g_cframes_studies, y_g_cframes_studies, Tx_data_cframes_studies, Ty_data_cframes_studies, cframes_labels, layers[n], 'layer', 'global', Rx_data_cframes_studies, 'cframes_studies_global_align')
    check_module_edges(x_g_velo_long_tracks, y_g_velo_long_tracks, Tx_data_velo_long_tracks, Ty_data_velo_long_tracks, velo_long_tracks_labels, layers[n], 'layer', 'global', Rx_data_velo_long_tracks, 'velo_long_tracks_global_align')
    check_module_edges(x_g_combi, y_g_combi, Tx_data_combi, Ty_data_combi, combi_labels, layers[n], 'layer', 'global', Rx_data_combi, 'combi_global_align')
    # do it for each individual datafile
    # all files
    make_edges_plot(x_g_v1, y_g_v1, Tx_data_v1, Ty_data_v1, legendlabels_v1, layers[n], 'layer', 'global', Rx_data_v1, 'v1_global_align', 'all')
    make_edges_plot(x_g_v2, y_g_v2, Tx_data_v2, Ty_data_v2, legendlabels_v2, layers[n], 'layer', 'global', Rx_data_v2, 'v2_global_align', 'all')
    make_edges_plot(x_g_v3, y_g_v3, Tx_data_v3, Ty_data_v3, legendlabels_v3, layers[n], 'layer', 'global', Rx_data_v3, 'v3_global_align', 'all')
    # individual
    make_edges_plot(x_g_v1, y_g_v1, Tx_data_v1, Ty_data_v1, legendlabels_v1, layers[n], 'layer', 'global', Rx_data_v1, 'v1_global_align', 'individual')
    make_edges_plot(x_g_v2, y_g_v2, Tx_data_v2, Ty_data_v2, legendlabels_v2, layers[n], 'layer', 'global', Rx_data_v2, 'v2_global_align', 'individual')
    make_edges_plot(x_g_v3, y_g_v3, Tx_data_v3, Ty_data_v3, legendlabels_v3, layers[n], 'layer', 'global', Rx_data_v3, 'v3_global_align', 'individual')
    # global plot for TxTzRxRz for 10 mu vs TxRz
    # make_3D_constants(tx_compxy, ty_compxy, tz_compxy, glob_x_compxy, glob_y_compxy, glob_z_compxy, labels_xy, layers[n])
plot_with_globals(Tx_v1, 'v1_glob_z_vs_local_', legendlabels_v1, layers, z_glob_v1, x_glob_v1, 'Tx')
plot_with_globals(Tx_v2, 'v2_glob_z_vs_local_', legendlabels_v2, layers, z_glob_v2, x_glob_v2, 'Tx')
plot_with_globals(Tx_v3, 'v3_glob_z_vs_local_', legendlabels_v3, layers, z_glob_v3, x_glob_v3, 'Tx')
plot_with_globals(Tx_retest, 'retest_glob_z_vs_local_', legendlabels_retest, layers, z_glob_retest, x_glob_retest, 'Tx')
plot_with_globals(Tx_v5, 'v5_glob_z_vs_local_', v5_labels, layers, z_glob_v5, x_glob_v5, 'Tx')
plot_with_globals(Tx_v6, 'v6_glob_z_vs_local_', v6_labels, layers, z_glob_v6, x_glob_v6, 'Tx')
plot_with_globals(Tx_v56, 'v56_comp_glob_z_vs_local_', v56_comp_labels, layers, z_glob_v56, x_glob_v56, 'Tx')
plot_with_globals(Tx_mod_and_halfmod, 'mod_and_halfmod_glob_z_vs_local_', mod_and_halfmod_labels, layers, z_glob_mod_and_halfmod, x_glob_mod_and_halfmod, 'Tx')
plot_with_globals(Tx_full_v1, 'full_v1_glob_z_vs_local_', full_v1_labels, layers, z_glob_full_v1, x_glob_full_v1, 'Tx')
plot_with_globals(Tx_new_conf, 'new_conf_glob_z_vs_local_', new_conf_labels, layers, z_glob_new_conf, x_glob_new_conf, 'Tx')
plot_with_globals(Tx_cframes_studies, 'cframes_studies_glob_z_vs_local_', cframes_labels, layers, z_glob_cframes_studies, x_glob_cframes_studies, 'Tx')
plot_with_globals(Tx_velo_long_tracks, 'velo_long_tracks_glob_z_vs_local_', velo_long_tracks_labels, layers, z_glob_velo_long_tracks, x_glob_velo_long_tracks, 'Tx')
plot_with_globals(Tx_combi, 'combi_glob_z_vs_local_', combi_labels, layers, z_glob_combi, x_glob_combi, 'Tx')

plot_with_globals(Ty_v1, 'v1_glob_z_vs_local_', legendlabels_v1, layers, z_glob_v1, x_glob_v1, 'Ty')
plot_with_globals(Ty_v2, 'v2_glob_z_vs_local_', legendlabels_v2, layers, z_glob_v2, x_glob_v2, 'Ty')
plot_with_globals(Ty_v3, 'v3_glob_z_vs_local_', legendlabels_v3, layers, z_glob_v3, x_glob_v3, 'Ty')
plot_with_globals(Ty_retest, 'retest_glob_z_vs_local_', legendlabels_retest, layers, z_glob_retest, x_glob_retest, 'Ty')
plot_with_globals(Ty_v5, 'v5_glob_z_vs_local_', v5_labels, layers, z_glob_v5, x_glob_v5, 'Ty')
plot_with_globals(Ty_v6, 'v6_glob_z_vs_local_', v6_labels, layers, z_glob_v6, x_glob_v6, 'Ty')
plot_with_globals(Ty_v56, 'v56_comp_glob_z_vs_local_', v56_comp_labels, layers, z_glob_v56, x_glob_v56, 'Ty')
plot_with_globals(Ty_mod_and_halfmod, 'mod_and_halfmod_glob_z_vs_local_', mod_and_halfmod_labels, layers, z_glob_mod_and_halfmod, x_glob_mod_and_halfmod, 'Ty')
plot_with_globals(Ty_full_v1, 'full_v1_glob_z_vs_local_', full_v1_labels, layers, z_glob_full_v1, x_glob_full_v1, 'Ty')
plot_with_globals(Ty_new_conf, 'new_conf_glob_z_vs_local_', new_conf_labels, layers, z_glob_new_conf, x_glob_new_conf, 'Ty')
plot_with_globals(Ty_cframes_studies, 'cframes_studies_glob_z_vs_local_', cframes_labels, layers, z_glob_cframes_studies, x_glob_cframes_studies, 'Ty')
plot_with_globals(Ty_velo_long_tracks, 'velo_long_tracks_glob_z_vs_local_', velo_long_tracks_labels, layers, z_glob_velo_long_tracks, x_glob_velo_long_tracks, 'Ty')
plot_with_globals(Ty_combi, 'combi_glob_z_vs_local_', combi_labels, layers, z_glob_combi, x_glob_combi, 'Ty')

plot_with_globals(Tz_v1, 'v1_glob_z_vs_local_', legendlabels_v1, layers, z_glob_v1, x_glob_v1, 'Tz')
plot_with_globals(Tz_v2, 'v2_glob_z_vs_local_', legendlabels_v2, layers, z_glob_v2, x_glob_v2, 'Tz')
plot_with_globals(Tz_v3, 'v3_glob_z_vs_local_', legendlabels_v3, layers, z_glob_v3, x_glob_v3, 'Tz')
plot_with_globals(Tz_retest, 'retest_glob_z_vs_local_', legendlabels_retest, layers, z_glob_retest, x_glob_retest, 'Tz')
plot_with_globals(Tz_v5, 'v5_glob_z_vs_local_', v5_labels, layers, z_glob_v5, x_glob_v5, 'Tz')
plot_with_globals(Tz_v6, 'v6_glob_z_vs_local_', v6_labels, layers, z_glob_v6, x_glob_v6, 'Tz')
plot_with_globals(Tz_v56, 'v56_comp_glob_z_vs_local_', v56_comp_labels, layers, z_glob_v56, x_glob_v56, 'Tz')
plot_with_globals(Tz_mod_and_halfmod, 'mod_and_halfmod_glob_z_vs_local_', mod_and_halfmod_labels, layers, z_glob_mod_and_halfmod, x_glob_mod_and_halfmod, 'Tz')
plot_with_globals(Tz_full_v1, 'full_v1_glob_z_vs_local_', full_v1_labels, layers, z_glob_full_v1, x_glob_full_v1, 'Tz')
plot_with_globals(Tz_new_conf, 'new_conf_glob_z_vs_local_', new_conf_labels, layers, z_glob_new_conf, x_glob_new_conf, 'Tz')
plot_with_globals(Tz_cframes_studies, 'cframes_studies_glob_z_vs_local_', cframes_labels, layers, z_glob_cframes_studies, x_glob_cframes_studies, 'Tz')
plot_with_globals(Tz_velo_long_tracks, 'velo_long_tracks_glob_z_vs_local_', velo_long_tracks_labels, layers, z_glob_velo_long_tracks, x_glob_velo_long_tracks, 'Tz')
plot_with_globals(Tz_combi, 'combi_glob_z_vs_local_', combi_labels, layers, z_glob_combi, x_glob_combi, 'Tz')

plot_with_globals(Rx_v1, 'v1_glob_z_vs_local_', legendlabels_v1, layers, z_glob_v1, x_glob_v1, 'Rx')
plot_with_globals(Rx_v2, 'v2_glob_z_vs_local_', legendlabels_v2, layers, z_glob_v2, x_glob_v2, 'Rx')
plot_with_globals(Rx_v3, 'v3_glob_z_vs_local_', legendlabels_v3, layers, z_glob_v3, x_glob_v3, 'Rx')
plot_with_globals(Rx_retest, 'retest_glob_z_vs_local_', legendlabels_retest, layers, z_glob_retest, x_glob_retest, 'Rx')
plot_with_globals(Rx_v5, 'v5_glob_z_vs_local_', v5_labels, layers, z_glob_v5, x_glob_v5, 'Rx')
plot_with_globals(Rx_v6, 'v6_glob_z_vs_local_', v6_labels, layers, z_glob_v6, x_glob_v6, 'Rx')
plot_with_globals(Rx_v56, 'v56_comp_glob_z_vs_local_', v56_comp_labels, layers, z_glob_v56, x_glob_v56, 'Rx')
plot_with_globals(Rx_mod_and_halfmod, 'mod_and_halfmod_glob_z_vs_local_', mod_and_halfmod_labels, layers, z_glob_mod_and_halfmod, x_glob_mod_and_halfmod, 'Rx')
plot_with_globals(Rx_full_v1, 'full_v1_glob_z_vs_local_', full_v1_labels, layers, z_glob_full_v1, x_glob_full_v1, 'Rx')
plot_with_globals(Rx_new_conf, 'new_conf_glob_z_vs_local_', new_conf_labels, layers, z_glob_new_conf, x_glob_new_conf, 'Rx')
plot_with_globals(Rx_cframes_studies, 'cframes_studies_glob_z_vs_local_', cframes_labels, layers, z_glob_cframes_studies, x_glob_cframes_studies, 'Rx')
plot_with_globals(Rx_velo_long_tracks, 'velo_long_tracks_glob_z_vs_local_', velo_long_tracks_labels, layers, z_glob_velo_long_tracks, x_glob_velo_long_tracks, 'Rx')
plot_with_globals(Rx_combi, 'combi_glob_z_vs_local_', combi_labels, layers, z_glob_combi, x_glob_combi, 'Rx')

plot_with_globals(Ry_v1, 'v1_glob_z_vs_local_', legendlabels_v1, layers, z_glob_v1, x_glob_v1, 'Ry')
plot_with_globals(Ry_v2, 'v2_glob_z_vs_local_', legendlabels_v2, layers, z_glob_v2, x_glob_v2, 'Ry')
plot_with_globals(Ry_v3, 'v3_glob_z_vs_local_', legendlabels_v3, layers, z_glob_v3, x_glob_v3, 'Ry')
plot_with_globals(Ry_retest, 'retest_glob_z_vs_local_', legendlabels_retest, layers, z_glob_retest, x_glob_retest, 'Ry')
plot_with_globals(Ry_v5, 'v5_glob_z_vs_local_', v5_labels, layers, z_glob_v5, x_glob_v5, 'Ry')
plot_with_globals(Ry_v6, 'v6_glob_z_vs_local_', v6_labels, layers, z_glob_v6, x_glob_v6, 'Ry')
plot_with_globals(Ry_v56, 'v56_comp_glob_z_vs_local_', v56_comp_labels, layers, z_glob_v56, x_glob_v56, 'Ry')
plot_with_globals(Ry_mod_and_halfmod, 'mod_and_halfmod_glob_z_vs_local_', mod_and_halfmod_labels, layers, z_glob_mod_and_halfmod, x_glob_mod_and_halfmod, 'Ry')
plot_with_globals(Ry_full_v1, 'full_v1_glob_z_vs_local_', full_v1_labels, layers, z_glob_full_v1, x_glob_full_v1, 'Ry')
plot_with_globals(Ry_new_conf, 'new_conf_glob_z_vs_local_', new_conf_labels, layers, z_glob_new_conf, x_glob_new_conf, 'Ry')
plot_with_globals(Ry_cframes_studies, 'cframes_studies_glob_z_vs_local_', cframes_labels, layers, z_glob_cframes_studies, x_glob_cframes_studies, 'Ry')
plot_with_globals(Ry_velo_long_tracks, 'velo_long_tracks_glob_z_vs_local_', velo_long_tracks_labels, layers, z_glob_velo_long_tracks, x_glob_velo_long_tracks, 'Ry')
plot_with_globals(Ry_combi, 'combi_glob_z_vs_local_', combi_labels, layers, z_glob_combi, x_glob_combi, 'Ry')

plot_with_globals(Rz_v1, 'v1_glob_z_vs_local_', legendlabels_v1, layers, z_glob_v1, x_glob_v1, 'Rz')
plot_with_globals(Rz_v2, 'v2_glob_z_vs_local_', legendlabels_v2, layers, z_glob_v2, x_glob_v2, 'Rz')
plot_with_globals(Rz_v3, 'v3_glob_z_vs_local_', legendlabels_v3, layers, z_glob_v3, x_glob_v3, 'Rz')
plot_with_globals(Rz_retest, 'retest_glob_z_vs_local_', legendlabels_retest, layers, z_glob_retest, x_glob_retest, 'Rz')
plot_with_globals(Rz_v5, 'v5_glob_z_vs_local_', v5_labels, layers, z_glob_v5, x_glob_v5, 'Rz')
plot_with_globals(Rz_v6, 'v6_glob_z_vs_local_', v6_labels, layers, z_glob_v6, x_glob_v6, 'Rz')
plot_with_globals(Rz_v56, 'v56_comp_glob_z_vs_local_', v56_comp_labels, layers, z_glob_v56, x_glob_v56, 'Rz')
plot_with_globals(Rz_mod_and_halfmod, 'mod_and_halfmod_glob_z_vs_local_', mod_and_halfmod_labels, layers, z_glob_mod_and_halfmod, x_glob_mod_and_halfmod, 'Rz')
plot_with_globals(Rz_full_v1, 'full_v1_glob_z_vs_local_', full_v1_labels, layers, z_glob_full_v1, x_glob_full_v1, 'Rz')
plot_with_globals(Rz_new_conf, 'new_conf_glob_z_vs_local_', new_conf_labels, layers, z_glob_new_conf, x_glob_new_conf, 'Rz')
plot_with_globals(Rz_cframes_studies, 'cframes_studies_glob_z_vs_local_', cframes_labels, layers, z_glob_cframes_studies, x_glob_cframes_studies, 'Rz')
plot_with_globals(Rz_velo_long_tracks, 'velo_long_tracks_glob_z_vs_local_', velo_long_tracks_labels, layers, z_glob_velo_long_tracks, x_glob_velo_long_tracks, 'Rz')
plot_with_globals(Rz_combi, 'combi_glob_z_vs_local_', combi_labels, layers, z_glob_combi, x_glob_combi, 'Rz')

glob_vs_glob(y_glob_v1, z_glob_v1, 'global_y', 'global_z', 'global_y_vs_global_z', legendlabels_v1)
