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
outname_prefix = 'SciFiAlignv3/'

def alt_modules(nums1, nums2, labels, ID, prefix):
    # print(nums1)
    outfiles = 'out_x_y_pos/'
    total_layer_num = 12 # number of layers
    total_num_runs = len(labels)
    x_data1 = nums1[0]
    x_data2 = nums1[1]
    y_data1 = nums2[0]
    y_data2 = nums2[1]

    L = ['Q2', 'Q3', 'Q0', 'Q1']

    x1_file1 = x_data1[0:5]  # Q0
    x2_file1 = x_data1[5:10]  # Q2
    x3_file1 = x_data1[10:15]  # Q1
    x4_file1 = x_data1[15:20]  # Q3

    y1_file1 = y_data1[0:5]  # Q0
    y2_file1 = y_data1[5:10]  # Q2
    y3_file1 = y_data1[10:15]  # Q1
    y4_file1 = y_data1[15:20]  # Q3

    x1_file2 = x_data2[0:5]  # Q0
    x2_file2 = x_data2[5:10]  # Q2
    x3_file2 = x_data2[10:15]  # Q1
    x4_file2 = x_data2[15:20]  # Q3

    y1_file2 = y_data2[0:5]  # Q0
    y2_file2 = y_data2[5:10]  # Q2
    y3_file2 = y_data2[10:15]  # Q1
    y4_file2 = y_data2[15:20]  # Q3

    plt.scatter(x_data1, y_data1, label = 'old')
    plt.scatter(x_data2, y_data2, label = '10 mu')
    plt.xticks(x_data1, ["Q0M0", "Q0M1", "Q0M2", "Q0M3", "Q0M4", "Q2M0", "Q2M1", "Q2M2", "Q2M3", "Q2M4", "Q1M0", "Q1M1", "Q1M2", "Q1M3", "Q1M4", "Q3M0", "Q3M1", "Q3M2", "Q3M3", "Q3M4"])
    plt.legend()
    plt.savefig('10mu_old_comp_' + prefix + '.pdf')
    plt.clf()

def make_full_layer_plot():
    fig, ax = plt.subplots()
    for module in range(num_modules_per_quarter): # top right
        ax.add_patch(Rectangle((bottom_left_corner[0] + ((module+1) * halfmodule_width), bottom_left_corner[1]), halfmodule_width, halfmodule_length, edgecolor='red', linewidth=2, fill=False))
        ax.set_xlim(-4000, 4000)
        ax.set_ylim(-3300, 3300)
    for module in range(num_modules_per_quarter): # top left
        ax.add_patch(Rectangle((bottom_left_corner[0] - (module * halfmodule_width), bottom_left_corner[1]), halfmodule_width, halfmodule_length, edgecolor='blue', linewidth=2, fill=False))
    for module in range(num_modules_per_quarter): # bottom right
        ax.add_patch(Rectangle((bottom_left_corner[0] + ((module+1) * halfmodule_width), bottom_left_corner[1] - halfmodule_length), halfmodule_width, halfmodule_length, edgecolor='green', linewidth=2, fill=False))
    for module in range(num_modules_per_quarter): # bottom left
        ax.add_patch(Rectangle((bottom_left_corner[0] - (module * halfmodule_width), bottom_left_corner[1] - halfmodule_length), halfmodule_width, halfmodule_length, edgecolor='black', linewidth=2, fill=False))
    plt.title('module fitting at the joint')
    plt.xlabel('global module position in x [mm]')
    plt.ylabel('global module position in y [mm]')
    plt.savefig('retest_uncertainty/out_rectangle/plot_global_positions.pdf')
    plt.clf()

def make_edges_plot(nums1, nums2, local1, local2, labels, ID, quarter_or_layer, local_or_global, x_rot, filenumbers='all'):
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
            print('y_top_min', y_top_min)
            print('y_top_max', y_top_max)
            print('y_bottom_min', y_bottom_min)
            print('y_bottom_max', y_bottom_max)
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
            plt.savefig(f'retest_uncertainty/out_rectangle/plot_edges_positions_run{num}_{filenumbers}_{ID}.pdf')
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
            print('y_top_min', y_top_min)
            print('y_top_max', y_top_max)
            print('y_bottom_min', y_bottom_min)
            print('y_bottom_max', y_bottom_max)
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
            plt.savefig(f'retest_uncertainty/out_rectangle/plot_edges_positions_{filenumbers}_{ID}_file{num}.pdf')
            # plt.clf()

def check_module_edges(nums1, nums2, local1, local2, labels, ID, quarter_or_layer, local_or_global, x_rot):
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
        plt.hlines(global_joint[1], x[0], x[9], 'red')
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
        plt.title(f'local translation')
        plt.savefig(f'{outname_prefix}{outfiles}' + 'module_edges_' + ID + '.pdf')
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

def plot_x_y_constants(nums1, nums2, local1, local2, labels, ID, quarter_or_layer, local_or_global):
    outfiles = 'out_x_y_pos/'
    total_layer_num = 12 # number of layers
    total_num_runs = len(labels)

    # x has 4 entries, 1 for each quarter, within these 4 we have as many as the number of input files
    x_Q0, x_Q2, x_Q1, x_Q3 = global_local_combiner(nums1, local1, quarter_or_layer, local_or_global)
    y_Q0, y_Q2, y_Q1, y_Q3 = global_local_combiner(nums2, local2, quarter_or_layer, local_or_global)

    L = ['Q2', 'Q3', 'Q0', 'Q1']

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
                plt.title(f'local translation')
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
        count += 1
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(f'{outname_prefix}{outfiles}' + 'x_vs_y_old_and_10mu_' + ID + '.pdf')
    plt.clf()

# def global_vs_local_plotting(input_data, global_variable_data, labels, dof_local, dof_global)

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
        '''
         split it into A and C side
        '''
        x_shifted = np.array(x_data)

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
            plt.plot(x_means[runs], z_positions[runs], ls='', marker=markers[runs], c=colors[runs], markersize=10, label=f'{run_labels[runs]}')
            if runs == 0:
                for i in range(total_layer_num):
                    plt.text(x_means[runs][i], z_positions[i]+((i/20) * (-1)**i), layers[i], fontsize=9)
            if plt.grid(True):
                plt.grid()
            plt.legend(loc='best')
            plt.xlabel(f'mean local x module position [mm]')
            plt.ylabel(f'global y module position [mm]')
            plt.title(f'mean {y_axis} vs. global y')
        plt.savefig(f'{outname_prefix}/{outfiles}' + 'all_runs_' + outname + '.pdf')
        plt.clf()
    if y_axis == 'Rx' or y_axis == 'Tx':
        correct_order = [2, 0, 1, 3, 6, 4, 5, 7, 10, 8, 9, 11]
        for runs in range(total_num_runs):

            correct_x_order = [x_means[runs][iter] for iter in correct_order]
            # print('correct_x_order', correct_x_order)
            # yerr=sem(x_means[runs]),
            size = 10
            if runs == 3:
                size = 11
            if runs == 5:
                size = 8
            plt.plot(z_positions, x_means[runs], ls='', marker=markers[runs], c=colors[runs], markersize=size, label=f'{run_labels[runs]}')
            if plt.grid(True):
                plt.grid()
            plt.legend(loc='best')
            plt.ylabel(f'mean {y_axis} module position [mm]')
            plt.title(f'mean position of layers in {y_axis} vs. global z position')
            plt.xticks(z_positions, ['T1U', 'T1V', 'T1X1', 'T1X2', 'T2U', 'T2V', 'T2X1', 'T2X2', 'T3U', 'T3V', 'T3X1', 'T3X2'], rotation=45, fontsize=10)
        plt.savefig(f'{outname_prefix}/{outfiles}' + 'all_runs_' + outname + '.pdf')
        plt.clf()
    if y_axis == 'Tz':
        correct_order = [2, 0, 1, 3, 6, 4, 5, 7, 10, 8, 9, 11]
        for runs in range(total_num_runs):

            correct_x_order = [x_means[runs][iter] for iter in correct_order]
            # print('correct_x_order', correct_x_order)
            # yerr=sem(x_means[runs]),
            # print('z pos local', x_means[runs])
            # print('z pos global', z_positions)
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


def plot(data_arr, survey_pos, outname, run_labels, title_label, layerID, outfiles=''):
    max_Q0, max_Q1, max_Q2, max_Q3 = [], [], [], [] # should store 2 values: value and where per layer
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
    for i in range(total_num_runs):
        # if len(survey_pos) == 0:
            # print('range index i =', i)
            # x1 = data_arr[i][0:5] - data_arr[i+1][0:5]  # Q0
            # x2 = data_arr[i][5:10] - data_arr[i+1][5:10]  # Q2
            # x3 = data_arr[i][10:15] - data_arr[i+1][10:15]  # Q1
            # x4 = data_arr[i][15:20] - data_arr[i+1][15:20]  # Q3
        if len(survey_pos) == 0:
            x1 = data_arr[i][0:5]  # Q0
            x2 = data_arr[i][5:10]  # Q2
            x3 = data_arr[i][10:15]  # Q1
            x4 = data_arr[i][15:20]  # Q3
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
                if "T" in title_label:
                    plt.ylabel(f'{title_label} [mm]')
                else:
                    plt.ylabel(f'{title_label} [mrad]')
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
                plt.xticks(x, ["M0", "M1", "M2", "M3", "M4"], rotation=45, fontsize=5)
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
    # new spacing
    # print('files:', files)
    if files[0] == 'retest_uncertainty/json/parsedlog_Tx_10micron_Rz_better.json' or files[0] == "retest_uncertainty/json/parsedlog_500k_old_unc_loose.json" or files[0] == "retest_uncertainty/json/parsedlog_2micron.json":
    # files[0] == "retest_uncertainty/json/parsedlog_500k_old_unc_loose.json":
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

def make_3D_constants(x_local, y_local, z_local, x_global, y_global, z_global, labels, ID):
    fig = plt.figure(figsize=plt.figaspect(1))
    n_files = len(labels)
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    # print(x_local)
    for i in range(n_files):
        ax.scatter(x_local[i], z_local[i], y_local[i], color=colors[i], label='local pos')
    ax.legend()
    ax.grid()
    ax.set_title('Q0M0: module positions vs hit positions')
    ax.set_xlabel('local x')
    ax.set_ylabel('local z')
    ax.set_zlabel('local y')

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    for i in range(n_files):
        ax.scatter(x_global[i], z_global[i], y_global[i], color=colors[i], label='global pos')
    ax.legend()
    ax.grid()
    ax.set_xlim(-2500, 2500)
    ax.set_ylim(-2500, 2500)
    ax.set_zlim(7000, 9000)
    ax.set_title('Q0M1: module positions vs hit positions')
    ax.set_xlabel('global x')
    ax.set_ylabel('global z')
    ax.set_zlabel('global y')

    plt.savefig('SciFiAlignv3/constants_diff_' + ID +'.pdf')
    # plt.show()

def make_outfiles(files, output_variables):
    align_outputs=[open_alignment(thisfile) for thisfile in files]
    plotted_alignables=[]
    for align_block in align_outputs:
        thislist=[]
        for key in align_block.keys():
            if "FT" in key:
                thislist.append(key)
        plotted_alignables.append(thislist)
    align_outputs=[convertGlobal(align_block,plotted_alignables[0]) for align_block in align_outputs]

    out_vars = []
    for var in output_variables:
        out_vars.append(get_data(files, var, align_outputs))
    # return align_outputs
    return out_vars

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
reduced_files = [\
         "align_logfiles_stability/json_files/parsedlog_256145.json",
         "align_logfiles_stability/json_files/parsedlog_256159.json",
         "align_logfiles_stability/json_files/parsedlog_256163.json",
         "align_logfiles_stability/json_files/parsedlog_256272.json",
         "align_logfiles_stability/json_files/parsedlog_256278.json",
         "align_logfiles_stability/json_files/parsedlog_256290.json",
]
reduced_labels = [\
        "256145-256159",
        "256159-256163",
        "256163-256272",
        "256272-256278",
        "256278-256290",
]
diff_labels = [\
            "255949-256030",
            "256030-256145",
            "256145-256159",
            "256159-256163",
            "256163-256272",
            "256272-256278",
            "256278-256290",
]

diff_md = [\
            "255949-256030",
            "256030-256145",
            "256145-256159",
            "256159-256163",
]

diff_mu = [\
            "256272-256278",
            "256278-256290",
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

# compare red to blue
f = [\
    "retest_uncertainty/json/parsedlog_500k_old_unc_loose.json",
    # "retest_uncertainty/json/parsedlog_500k_tuned_unc_loose.json",
    # "retest_uncertainty/json/parsedlog_0_0001_Tx_retune.json",
    # "retest_uncertainty/json/parsedlog_baseTx_tune_rest.json",
    # "retest_uncertainty/json/parsedlog_0_8_onlyTx.json",
    # "retest_uncertainty/json/parsedlog_0_6_onlyTx.json",

    # "retest_uncertainty/json/parsedlog_2micron.json",
    "retest_uncertainty/json/parsedlog_Tx_10micron_Rz_better.json",
    "retest_uncertainty/json/parsedlog_TxTzRxRz_iter4_10micron_tuned.json",
    # "retest_uncertainty/json/parsedlog_100mu_TxTzRxRz_iter15.json",
    # "retest_uncertainty/json/parsedlog_100mu_TxRz.json",
    # "retest_uncertainty/json/parsedlog_small_joint_Rx_10mu_Tx_TxRz.json",
    "retest_uncertainty/json/parsedlog_TxRxRz_smallRxSurveyUnc.json",
    # "retest_uncertainty/json/parsedlog_RxJoints_0.json",
    "retest_uncertainty/json/parsedlog_v2_fix_survey.json",
    "retest_uncertainty/json/parsedlog_fix_V_layers.json",
    'retest_uncertainty/json/parsedlog_global_TxTzRxRz.json',
    'retest_uncertainty/json/parsedlog_wouter_constraint.json'
]
lab = [\
    # "old",
    # "new",
    # "0_0001_tx",
    # "baseTx_tuned",
    # "0_8Tx",
    # "0_6Tx",

    'V9_old_joint_config',
    'TxRz_10mu',
    "TxTzRxRz_10mu",
    # "100mu_TxTzRxRz",
    # "100mu_TxRz",
    # "10mu_smallRxJoint",
    "10mu_TxRxRz_smallRxSurveyUnc",
    # "10mu_RxJoints_small",
    '10mu_TxRxRz_T2V_fixed',
    "10mu_TxRxRz_T1U_T2V",
    '10mu_TxTzRxRz_with_globModules',
    'constraint_wouter'
]

xy_comp_input = [\
    "retest_uncertainty/json/parsedlog_500k_old_unc_loose.json",
    "retest_uncertainty/json/parsedlog_Tx_10micron_Rz_better.json",
    "retest_uncertainty/json/parsedlog_TxTzRxRz_iter4_10micron_tuned.json",
    # "retest_uncertainty/json/parsedlog_100mu_TxTzRxRz_iter15.json",
    # "retest_uncertainty/json/parsedlog_100mu_TxRz.json",
    # "retest_uncertainty/json/parsedlog_small_joint_Rx_10mu_Tx_TxRz.json",
    "retest_uncertainty/json/parsedlog_TxRxRz_smallRxSurveyUnc.json",
    # "retest_uncertainty/json/parsedlog_RxJoints_0.json",
    "retest_uncertainty/json/parsedlog_v2_fix_survey.json",
    "retest_uncertainty/json/parsedlog_fix_V_layers.json",
    'retest_uncertainty/json/parsedlog_global_TxTzRxRz.json',
    # 'retest_uncertainty/json/parsedlog_wouter_constraint.json'
]
labels_xy = [\
    'V9_old_joint_config',
    'TxRz_10mu',
    "TxTzRxRz_10mu",
    # "100mu_TxTzRxRz",
    # "100mu_TxRz",
    # "10mu_smallRxJoint",
    "10mu_TxRxRz_smallRxSurveyUnc",
    # "10mu_RxJoints_small",
    '10mu_TxRxRz_T2V_fixed',
    "10mu_TxRxRz_T1U_T2V",
    '10mu_TxTzRxRz_with_globModules',
    # 'constraint_wouter'
]

# plot constants of only Tx tuning for strict particles
pre_fix = [\
    "retest_uncertainty/json/parsedlog_500k_old_unc_loose.json",
    "retest_uncertainty/json/parsedlog_pre_0_1.json",
    "retest_uncertainty/json/parsedlog_pre_0_15.json",
    "retest_uncertainty/json/parsedlog_pre_0_2.json",
    "retest_uncertainty/json/parsedlog_pre_0_25.json",
    "retest_uncertainty/json/parsedlog_pre_0_3.json",
    "retest_uncertainty/json/parsedlog_pre_0_4.json",
    "retest_uncertainty/json/parsedlog_pre_0_45.json",
    "retest_uncertainty/json/parsedlog_pre_0_5.json",
]
pre_fix_labels = [\
    "base_500k",
    "0_1",
    "0_15",
    "0_2",
    "0_25",
    "0_3",
    "0_4",
    "0_45",
    "0_5",
]

survey_module_positions = 'survey/survey_Modules.yml'

layers = ['T1U', 'T1V', 'T1X1', 'T1X2', 'T2U', 'T2V', 'T2X1', 'T2X2', 'T3U', 'T3V', 'T3X1', 'T3X2']

runs = ["T3UHL0Q0M0", "T3UHL0Q0M1", "T3UHL0Q0M2", "T3UHL0Q0M3", "T3UHL0Q0M4", "T3UHL0Q2M0", "T3UHL0Q2M1", "T3UHL0Q2M2", "T3UHL0Q2M3", "T3UHL0Q2M4", "T3UHL1Q1M0"#\
        , "T3UHL1Q1M1", "T3UHL1Q1M2", "T3UHL1Q1M3", "T3UHL1Q1M4", "T3UHL1Q3M0", "T3UHL1Q3M1", "T3UHL1Q3M2", "T3UHL1Q3M3", "T3UHL1Q3M4"]

plotting_variables = ['Tx', 'Ty', 'Tz', 'nHits', 'nTracks', 'x_global', 'y_global', 'z_global']
align_outputs = make_outfiles(files, plotting_variables)
align_outputs_md = make_outfiles(files_md, plotting_variables)
align_outputs_mu = make_outfiles(files_mu, plotting_variables)
align_outputs_red = make_outfiles(reduced_files, plotting_variables)
print('a')
pref_fix_outputs = make_outfiles(pre_fix, plotting_variables)
print('b')
print('c')
red_blue_vars = ['Tx', 'Ty', 'Tz', 'Rx', 'Ry', 'Rz', 'nHits', 'nTracks', 'x_global', 'y_global', 'z_global']
align_outputs_xy_diff = make_outfiles(xy_comp_input, red_blue_vars)
red_blue_f = make_outfiles(f, red_blue_vars)

# for all files
tx      = align_outputs[0]
ty      = align_outputs[1]
tz      = align_outputs[2]
nHits   = align_outputs[3]
nTracks = align_outputs[4]
x_glob  = align_outputs[5]
y_glob  = align_outputs[6]
z_glob  = align_outputs[7]

survey_Tx = get_survey_data(survey_module_positions, 'Tx', 'position')
survey_Ty = get_survey_data(survey_module_positions, 'Ty', 'position')
survey_Tz = get_survey_data(survey_module_positions, 'Tz', 'position')
survey_Rz = get_survey_data(survey_module_positions, 'Rz', 'rotation')

# for magnet down
tx_md      = align_outputs_md[0]
ty_md      = align_outputs_md[1]
tz_md      = align_outputs_md[2]
nHits_md   = align_outputs_md[3]
nTracks_md = align_outputs_md[4]
x_glob_md  = align_outputs_md[5]
y_glob_md  = align_outputs_md[6]
z_glob_md  = align_outputs_md[7]

# for magnet up
tx_mu      = align_outputs_mu[0]
ty_mu      = align_outputs_mu[1]
tz_mu      = align_outputs_mu[2]
nHits_mu   = align_outputs_mu[3]
nTracks_mu = align_outputs_mu[4]
x_glob_mu  = align_outputs_mu[5]
y_glob_mu  = align_outputs_mu[6]
z_glob_mu  = align_outputs_mu[7]

# reduced data
tx_red      = align_outputs_red[0]
ty_red      = align_outputs_red[1]
tz_red      = align_outputs_red[2]
nHits_red   = align_outputs_red[3]
nTracks_red = align_outputs_red[4]
x_glob_red  = align_outputs_red[5]
y_glob_red  = align_outputs_red[6]
z_glob_red  = align_outputs_red[7]

# red blue
tx_rb      = red_blue_f[0]
ty_rb      = red_blue_f[1]
tz_rb      = red_blue_f[2]
rx_rb      = red_blue_f[3]
ry_rb      = red_blue_f[4]
rz_rb      = red_blue_f[5]
nHits_rb   = red_blue_f[6]
nTracks_rb = red_blue_f[7]
x_glob_rb  = red_blue_f[8]
y_glob_rb  = red_blue_f[9]
z_glob_rb  = red_blue_f[10]

# only Tx tuning
tx_pre      = pref_fix_outputs[0]
ty_pre      = pref_fix_outputs[1]
tz_pre      = pref_fix_outputs[2]
nHits_pre   = pref_fix_outputs[3]
nTracks_pre = pref_fix_outputs[4]
x_glob_pre  = pref_fix_outputs[5]
y_glob_pre  = pref_fix_outputs[6]
z_glob_pre  = pref_fix_outputs[7]

tx_xydiff      = align_outputs_xy_diff[0]
ty_xydiff      = align_outputs_xy_diff[1]
tz_xydiff      = align_outputs_xy_diff[2]
rx_xydiff      = align_outputs_xy_diff[3]
ry_xydiff      = align_outputs_xy_diff[4]
rz_xydiff      = align_outputs_xy_diff[5]
nhits_xydiff   = align_outputs_xy_diff[6]
ntracks_xydiff = align_outputs_xy_diff[7]
glob_x_xydiff  = align_outputs_xy_diff[8]
glob_y_xydiff  = align_outputs_xy_diff[9]
glob_z_xydiff  = align_outputs_xy_diff[10]

for n in range(12):
    tx_data = tx[n]
    ty_data = ty[n]
    tz_data = tz[n]
    x_g = x_glob[n]
    y_g = y_glob[n]
    z_g = z_glob[n]
    nHits_data = nHits[n]
    nTracks_data = nTracks[n]
    # plots the frontview quarter plots
    # plot(tx_data, survey_Tx, 'diff_data', legendlabels, 'Tx', layers[n])  # set [] to survey Tx if i want to compare to survey positions
    # plot(tz_data, [], 'diff_data', legendlabels, 'Tz', layers[n])
    plot(tx_data, [], 'diff_data', diff_labels, 'Tx', layers[n])  # set [] to survey Tx if i want to compare to survey positions
    # plot(tz_data, [], 'data_diff', diff_labels, 'Tz', layers[n])

    # plot(nHits_data, [], 'n_Hits', legendlabels, 'nHits', layers[n])
    # plot(chi2, 'chi2', legendlabels, 'localDeltaChi2', layers[n])

    # top view plots
    # compare_alignments(tx_data, 'diff_runs', legendlabels, 'Tx', layers[n])
    # compare_alignments(nHits_data, 'nHits_diff', legendlabels, 'nHits', layers[n])
    # compare_alignments(nTracks_data, 'nTracks_diff', legendlabels, 'nTracks', layers[n])
    # compare_alignments(chi2, 'chi2', z_g, legendlabels, 'localDeltaChi2', layers[n])

    # now for md only
    tx_data_md = tx_md[n]
    ty_data_md = ty_md[n]
    tz_data_md = tz_md[n]
    x_g_md = x_glob_md[n]
    y_g_md = y_glob_md[n]
    z_g_md = z_glob_md[n]
    # plot(tx_data_md, [], 'diff_MD', diff_md, 'Tx', layers[n])  # set [] to survey Tx if i want to compare to survey positions
    # plot(tz_data_md, survey_Tx, 'diff_MD', legendlabels_md, 'Tz', layers[n])

    # now for mu only
    tx_data_mu = tx_mu[n]
    ty_data_mu = ty_mu[n]
    tz_data_mu = tz_mu[n]
    x_g_mu = x_glob_mu[n]
    y_g_mu = y_glob_mu[n]
    z_g_mu = z_glob_mu[n]

    # plot(tx_data_mu, [], 'diff_MU', diff_mu, 'Tx', layers[n])  # set [] to survey Tx if i want to compare to survey positions
    # plot(tz_data_mu, survey_Tz, 'diff_MU', diff_mu, 'Tz', layers[n])

    tx_data_red = tx_red[n]
    ty_data_red = ty_red[n]
    tz_data_red = tz_red[n]
    x_g_data_red = x_glob_red[n]
    y_g_data_red = y_glob_red[n]
    z_g_data_red = z_glob_red[n]

    # plot(tx_data_red, [], 'diff_reduced_Tx', reduced_labels, 'Tx', layers[n])  # set [] to survey Tx if i want to compare to survey positions
    # plot(tz_data_red, [], 'diff_reduced_Tz', reduced_labels, 'Tz', layers[n])

    tx_d_rb = tx_rb[n]
    ty_d_rb = ty_rb[n]
    tz_d_rb = tz_rb[n]
    rx_d_rb = rx_rb[n]
    ry_d_rb = ry_rb[n]
    rz_d_rb = rz_rb[n]
    x_g_d_rb = x_glob_rb[n]
    y_g_d_rb = y_glob_rb[n]
    z_g_d_rb = z_glob_rb[n]

    plot(tx_d_rb, [], 'blue_red_comp', lab, 'Tx', layers[n])  # set [] to survey Tx if i want to compare to survey positions
    plot(ty_d_rb, [], 'blue_red_comp', lab, 'Ty', layers[n])  # set [] to survey Tx if i want to compare to survey positions
    plot(tz_d_rb, [], 'blue_red_comp', lab, 'Tz', layers[n])  # set [] to survey Tx if i want to compare to survey positions
    plot(rx_d_rb, [], 'blue_red_comp', lab, 'Rx', layers[n])  # set [] to survey Tx if i want to compare to survey positions
    plot(ry_d_rb, [], 'blue_red_comp', lab, 'Ry', layers[n])  # set [] to survey Tx if i want to compare to survey positions
    plot(rz_d_rb, [], 'blue_red_comp', lab, 'Rz', layers[n])  # set [] to survey Tx if i want to compare to survey positions

    tx_pre_d = tx_pre[n]
    ty_pre_d = ty_pre[n]
    tz_pre_d = tz_pre[n]
    x_g_pre_d = x_glob_pre[n]
    y_g_pre_d = y_glob_pre[n]
    z_g_pre_d = z_glob_pre[n]
    plot(tx_pre_d, [], 'onlyTx_comp_base', pre_fix_labels, 'Tx', layers[n], 'constants_Tx')
    # compare x y distribution
    tx_compxy = tx_xydiff[n]
    ty_compxy = ty_xydiff[n]
    tz_compxy = tz_xydiff[n]
    rx_compxy = rx_xydiff[n]
    ry_compxy = ry_xydiff[n]
    rz_compxy = rz_xydiff[n]
    nhits_compxy = nhits_xydiff[n]
    ntracks_compxy = ntracks_xydiff[n]
    glob_x_compxy = glob_x_xydiff[n]
    glob_y_compxy = glob_y_xydiff[n]
    glob_z_compxy = glob_z_xydiff[n]
    # print(len(glob_z_compxy), len(glob_z_compxy[0]))

    plot_x_y_constants(glob_x_compxy, glob_y_compxy, tx_compxy, ty_compxy, labels_xy, layers[n], 'quarter', 'global')
    check_module_edges(glob_x_compxy, glob_y_compxy, tx_compxy, ty_compxy, labels_xy, layers[n], 'layer', 'global', rx_compxy)
    # do it for each individual datafile
    # all files
    make_edges_plot(glob_x_compxy, glob_y_compxy, tx_compxy, ty_compxy, labels_xy, layers[n], 'layer', 'global', rx_compxy, 'all')
    # individual
    make_edges_plot(glob_x_compxy, glob_y_compxy, tx_compxy, ty_compxy, labels_xy, layers[n], 'layer', 'global', rx_compxy, 'individual')
    # global plot for TxTzRxRz for 10 mu vs TxRz
    # make_3D_constants(tx_compxy, ty_compxy, tz_compxy, glob_x_compxy, glob_y_compxy, glob_z_compxy, labels_xy, layers[n])
plot_with_globals(tx_xydiff, 'glob_z_vs_local_Tx', labels_xy, layers, glob_z_xydiff, glob_x_xydiff, 'Tx')
plot_with_globals(rx_xydiff, 'glob_z_vs_local_Rx', labels_xy, layers, glob_z_xydiff, glob_x_xydiff, 'Rx')
plot_with_globals(tz_xydiff, 'glob_z_vs_local_Tz', labels_xy, layers, glob_z_xydiff, glob_x_xydiff, 'Tz')

glob_vs_glob(glob_y_xydiff, glob_z_xydiff, 'global_y', 'global_z', 'global_y_vs_global_z', labels_xy)
