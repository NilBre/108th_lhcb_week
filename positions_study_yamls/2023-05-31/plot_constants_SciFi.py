#import ROOT
#from ROOT import TFile, TTree, gROOT, AddressOf, RDataFrame
from math import *
import sys, os
import numpy as np
import random
import yaml
import io
import matplotlib
import matplotlib.pyplot as plt
import glob
from grepfunc import grep_iter
import mplhep as hep
import numpy as np
from termcolor import colored
import argparse
import json
import re
#matplotlib.rcParams['text.usetex'] = True
#matplotlib.rcParams['text.latex.preamble'] = r"\usepackage{amsmath} \usepackage{amssymb}"

#T1: UVX1X2
#T2: UVX1X2
#T3: UVX1X2

def get_positions(files):
    num_files = len(files)
    iter_num = 0

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

    T1U_Tx_yml = [[] for _ in range(num_files)]
    T1U_Tx = [[] for _ in range(num_files)]
    T1V_Tx_yml = [[] for _ in range(num_files)]
    T1X1_Tx_yml = [[] for _ in range(num_files)]
    T1V_Tx = [[] for _ in range(num_files)]
    T1X1_Tx = [[] for _ in range(num_files)]
    T1X2_Tx_yml = [[] for _ in range(num_files)]
    T1X2_Tx = [[] for _ in range(num_files)]

    T2U_Tx_yml = [[] for _ in range(num_files)]
    T2U_Tx = [[] for _ in range(num_files)]
    T2V_Tx_yml = [[] for _ in range(num_files)]
    T2X1_Tx_yml = [[] for _ in range(num_files)]
    T2V_Tx = [[] for _ in range(num_files)]
    T2X1_Tx = [[] for _ in range(num_files)]
    T2X2_Tx_yml = [[] for _ in range(num_files)]
    T2X2_Tx = [[] for _ in range(num_files)]

    T3U_Tx_yml = [[] for _ in range(num_files)]
    T3U_Tx = [[] for _ in range(num_files)]
    T3V_Tx_yml = [[] for _ in range(num_files)]
    T3X1_Tx_yml = [[] for _ in range(num_files)]
    T3V_Tx = [[] for _ in range(num_files)]
    T3X1_Tx = [[] for _ in range(num_files)]
    T3X2_Tx_yml = [[] for _ in range(num_files)]
    T3X2_Tx = [[] for _ in range(num_files)]

    print('all files:', files)
    for file in files:
        print('#####################')
        print(f'iteration', iter_num + 1)
        print('current file:', file)
        runs_T1 = ["T1UHL0Q0M0", "T1UHL0Q0M1", "T1UHL0Q0M2", "T1UHL0Q0M3", "T1UHL0Q0M4", "T1UHL0Q2M0", "T1UHL0Q2M1", "T1UHL0Q2M2", "T1UHL0Q2M3", "T1UHL0Q2M4", "T1UHL1Q1M0", "T1UHL1Q1M1", "T1UHL1Q1M2", "T1UHL1Q1M3", "T1UHL1Q1M4", "T1UHL1Q3M0", "T1UHL1Q3M1", "T1UHL1Q3M2", "T1UHL1Q3M3", "T1UHL1Q3M4"]

        runs_T2 = ["T2UHL0Q0M0", "T2UHL0Q0M1", "T2UHL0Q0M2", "T2UHL0Q0M3", "T2UHL0Q0M4", "T2UHL0Q2M0", "T2UHL0Q2M1", "T2UHL0Q2M2", "T2UHL0Q2M3", "T2UHL0Q2M4", "T2UHL1Q1M0",
        "T2UHL1Q1M1", "T2UHL1Q1M2", "T2UHL1Q1M3", "T2UHL1Q1M4", "T2UHL1Q3M0", "T2UHL1Q3M1", "T2UHL1Q3M2", "T2UHL1Q3M3", "T2UHL1Q3M4"]

        runs = ["T3UHL0Q0M0", "T3UHL0Q0M1", "T3UHL0Q0M2", "T3UHL0Q0M3", "T3UHL0Q0M4", "T3UHL0Q2M0", "T3UHL0Q2M1", "T3UHL0Q2M2", "T3UHL0Q2M3", "T3UHL0Q2M4", "T3UHL1Q1M0"#\
        , "T3UHL1Q1M1", "T3UHL1Q1M2", "T3UHL1Q1M3", "T3UHL1Q1M4", "T3UHL1Q3M0", "T3UHL1Q3M1", "T3UHL1Q3M2", "T3UHL1Q3M3", "T3UHL1Q3M4"]
        # print(len(runs_T1), len(runs_T2), len(runs))
        x = list(range(len(runs)))
        for j in range(0,len(stations)):
           for k in range(0,len(layers)):
              if j==0 and k==0:
                 #string = runs[j]
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
        # print(runs_T3_U)
        for i in range(0,len(runs)):
           with open(file, 'r') as stream:
              data_loaded = yaml.load(stream, Loader=yaml.Loader)
              # T1
              # print('rotations, all:', data_loaded[runs_T1_U[iter_num][i]]['rotation'])
              # print('rotations, Rz:', data_loaded[runs_T1_U[iter_num][i]]['rotation'][2])
              # print(T1U_Tx_yml[iter_num][i])
              T1U_Tx_yml[iter_num].append(data_loaded[runs_T1_U[iter_num][i]]['rotation'][2])
              T1U_Tx[iter_num].append(float(re.findall(r'\d+',T1U_Tx_yml[iter_num][i])[0] + "." + re.findall(r'\d+',T1U_Tx_yml[iter_num][i])[1]))

              T1V_Tx_yml[iter_num].append(data_loaded[runs_T1_V[iter_num][i]]['rotation'][2])
              T1V_Tx[iter_num].append(float(re.findall(r'\d+',T1V_Tx_yml[iter_num][i])[0] + "." + re.findall(r'\d+',T1V_Tx_yml[iter_num][i])[1]))

              T1X1_Tx_yml[iter_num].append(data_loaded[runs_T1_X1[iter_num][i]]['rotation'][2])
              T1X1_Tx[iter_num].append(float(re.findall(r'\d+',T1X1_Tx_yml[iter_num][i])[0] + "." + re.findall(r'\d+',T1X1_Tx_yml[iter_num][i])[1]))

              T1X2_Tx_yml[iter_num].append(data_loaded[runs_T1_X2[iter_num][i]]['rotation'][2])
              T1X2_Tx[iter_num].append(float(re.findall(r'\d+',T1X2_Tx_yml[iter_num][i])[0] + "." + re.findall(r'\d+',T1X2_Tx_yml[iter_num][i])[1]))

              # T2
              T2U_Tx_yml[iter_num].append(data_loaded[runs_T2_U[iter_num][i]]['rotation'][2])
              T2U_Tx[iter_num].append(float(re.findall(r'\d+',T2U_Tx_yml[iter_num][i])[0] + "." + re.findall(r'\d+',T2U_Tx_yml[iter_num][i])[1]))

              T2V_Tx_yml[iter_num].append(data_loaded[runs_T2_V[iter_num][i]]['rotation'][2])
              T2V_Tx[iter_num].append(float(re.findall(r'\d+',T2V_Tx_yml[iter_num][i])[0] + "." + re.findall(r'\d+',T2V_Tx_yml[iter_num][i])[1]))

              T2X1_Tx_yml[iter_num].append(data_loaded[runs_T2_X1[iter_num][i]]['rotation'][2])
              T2X1_Tx[iter_num].append(float(re.findall(r'\d+',T2X1_Tx_yml[iter_num][i])[0] + "." + re.findall(r'\d+',T2X1_Tx_yml[iter_num][i])[1]))

              T2X2_Tx_yml[iter_num].append(data_loaded[runs_T2_X2[iter_num][i]]['rotation'][2])
              T2X2_Tx[iter_num].append(float(re.findall(r'\d+',T2X2_Tx_yml[iter_num][i])[0] + "." + re.findall(r'\d+',T2X2_Tx_yml[iter_num][i])[1]))

              # T3
              T3U_Tx_yml[iter_num].append(data_loaded[runs_T3_U[iter_num][i]]['rotation'][2])
              T3U_Tx[iter_num].append(float(re.findall(r'\d+',T3U_Tx_yml[iter_num][i])[0] + "." + re.findall(r'\d+',T3U_Tx_yml[iter_num][i])[1]))

              T3V_Tx_yml[iter_num].append(data_loaded[runs_T3_V[iter_num][i]]['rotation'][2])
              T3V_Tx[iter_num].append(float(re.findall(r'\d+',T3V_Tx_yml[iter_num][i])[0] + "." + re.findall(r'\d+',T3V_Tx_yml[iter_num][i])[1]))

              T3X1_Tx_yml[iter_num].append(data_loaded[runs_T3_X1[iter_num][i]]['rotation'][2])
              T3X1_Tx[iter_num].append(float(re.findall(r'\d+',T3X1_Tx_yml[iter_num][i])[0] + "." + re.findall(r'\d+',T3X1_Tx_yml[iter_num][i])[1]))

              T3X2_Tx_yml[iter_num].append(data_loaded[runs_T3_X2[iter_num][i]]['rotation'][2])
              T3X2_Tx[iter_num].append(float(re.findall(r'\d+',T3X2_Tx_yml[iter_num][i])[0] + "." + re.findall(r'\d+',T3X2_Tx_yml[iter_num][i])[1]))
        iter_num += 1
    return np.array(T1U_Tx), np.array(T1V_Tx), np.array(T1X1_Tx), np.array(T1X2_Tx), np.array(T2U_Tx), np.array(T2V_Tx), np.array(T2X1_Tx), np.array(T2X2_Tx), np.array(T3U_Tx), np.array(T3V_Tx), np.array(T3X1_Tx), np.array(T3X2_Tx)

if __name__ == '__main__':
    hep.style.use(hep.style.LHCb2)
    import matplotlib.patches as mpl_patches

    def meta_constructor(loader, node):
       return loader.construct_mapping(node)

    yaml.add_constructor('!alignment', meta_constructor)

    stations = ["T1", "T2", "T3"]
    layers = ["U", "V", "X1", "X2"]

    # folder on pc
    path_run_folder = "/mnt/c/Users/Nils/Desktop/Promotion/SciFi/positions_study_yamls/2023-05-31"

    magUp_yaml_files = [path_run_folder + '/256290/Modules.yml']
    magDown_yaml_files = [
                        path_run_folder + "/256163/Modules.yml",
                        path_run_folder + "/256159/Modules.yml",
                        path_run_folder + "/256145/Modules.yml",
                        path_run_folder + "/256030/Modules.yml",
    ]
    md_runs = ["/256163/Modules.yml", "/256159/Modules.yml", "/256145/Modules.yml", "/256030/Modules.yml"]
    mu_runs = ['/256290/Modules.yml']
    md_vs_mu = [path_run_folder + '/256290/Modules.yml', path_run_folder + "/256163/Modules.yml"]

    runs_T1 = ["T1UHL0Q0M0", "T1UHL0Q0M1", "T1UHL0Q0M2", "T1UHL0Q0M3", "T1UHL0Q0M4", "T1UHL0Q2M0", "T1UHL0Q2M1", "T1UHL0Q2M2", "T1UHL0Q2M3", "T1UHL0Q2M4", "T1UHL1Q1M0",
    "T1UHL1Q1M1", "T1UHL1Q1M2", "T1UHL1Q1M3", "T1UHL1Q1M4", "T1UHL1Q3M0", "T1UHL1Q3M1", "T1UHL1Q3M2", "T1UHL1Q3M3", "T1UHL1Q3M4"]

    runs_T2 = ["T2UHL0Q0M0", "T2UHL0Q0M1", "T2UHL0Q0M2", "T2UHL0Q0M3", "T2UHL0Q0M4", "T2UHL0Q2M0", "T2UHL0Q2M1", "T2UHL0Q2M2", "T2UHL0Q2M3", "T2UHL0Q2M4", "T2UHL1Q1M0",
    "T2UHL1Q1M1", "T2UHL1Q1M2", "T2UHL1Q1M3", "T2UHL1Q1M4", "T2UHL1Q3M0", "T2UHL1Q3M1", "T2UHL1Q3M2", "T2UHL1Q3M3", "T2UHL1Q3M4"]

    runs = ["T3UHL0Q0M0", "T3UHL0Q0M1", "T3UHL0Q0M2", "T3UHL0Q0M3", "T3UHL0Q0M4", "T3UHL0Q2M0", "T3UHL0Q2M1", "T3UHL0Q2M2", "T3UHL0Q2M3", "T3UHL0Q2M4", "T3UHL1Q1M0"#\
    , "T3UHL1Q1M1", "T3UHL1Q1M2", "T3UHL1Q1M3", "T3UHL1Q1M4", "T3UHL1Q3M0", "T3UHL1Q3M1", "T3UHL1Q3M2", "T3UHL1Q3M3", "T3UHL1Q3M4"]
    # list of object names for all layers

    # T1
    runs_V_T1 = []
    for i in range(0,len(runs_T1)):
       string = runs_T1[i]
       runs_V_T1.append(string.replace("U", "V"))

    runs_X1_T1 = []
    for i in range(0,len(runs_T1)):
       string = runs_T1[i]
       runs_X1_T1.append(string.replace("U", "X1"))

    runs_X2_T1 = []
    for i in range(0,len(runs_T1)):
       string = runs_T1[i]
       runs_X2_T1.append(string.replace("U", "X2"))

    # T2
    runs_V_T2 = []
    for i in range(0,len(runs_T2)):
       string = runs_T2[i]
       runs_V_T2.append(string.replace("U", "V"))

    runs_X1_T2 = []
    for i in range(0,len(runs_T2)):
       string = runs_T2[i]
       runs_X1_T2.append(string.replace("U", "X1"))

    runs_X2_T2 = []
    for i in range(0,len(runs_T2)):
       string = runs_T2[i]
       runs_X2_T2.append(string.replace("U", "X2"))

    # T3
    runs_V_T3 = []
    for i in range(0,len(runs)):
       string = runs[i]
       runs_V_T3.append(string.replace("U", "V"))

    runs_X1_T3 = []
    for i in range(0,len(runs)):
       string = runs[i]
       runs_X1_T3.append(string.replace("U", "X1"))

    runs_X2_T3 = []
    for i in range(0,len(runs)):
       string = runs[i]
       runs_X2_T3.append(string.replace("U", "X2"))

    x = list(range(len(runs)))

    # extract run numbers from file names
    mu_run_numbers, md_run_numbers = [], []
    for file in md_runs:
        num = re.findall(r'\d+', file)
        md_run_numbers.append(num[0])
    for file in mu_runs:
        num = re.findall(r'\d+', file)
        mu_run_numbers.append(num[0])

    outdir = 'outfiles'
    pos = get_positions(magDown_yaml_files)

    # plot T2X2 for full magDown
    fig, ax = plt.subplots(1,1, figsize=(10,10))
    r1 = 215/255
    g1 = 48/255
    b1 = 39/255
    r2 = 252/255
    g2 = 141/255
    b2 = 89/255

    #ax.scatter = (np.array(runs),np.array(VPRight_Tx))

    run_labels = [runs_T1, runs_V_T1, runs_X1_T1, runs_X2_T1, runs_T2, runs_V_T2, runs_X1_T2, runs_X2_T2, runs, runs_V_T3, runs_X1_T3, runs_X2_T3]
    out_labels = ['T1U', 'T1V', 'T1X1', 'T1X2', 'T2U', 'T2V', 'T2X1', 'T2X2', 'T3U', 'T3V', 'T3X1', 'T3X2']
    for i in range(12):
        plt.xticks(x, run_labels[i])
        colors = ['black', 'red', 'blue', 'green']

        # this is the fix for Tz
        # the values with 1e-5 magically change to +00
        vals_163 = abs(pos[i][0])
        vals_159 = abs(pos[i][1])
        vals_145 = abs(pos[i][2])
        vals_030 = abs(pos[i][3])
        vals_163 = [i * 1e3 if i < 1 else i * 1e-2 for i in vals_163]
        vals_159 = [i * 1e3 if i < 1 else i * 1e-2 for i in vals_159]
        vals_145 = [i * 1e3 if i < 1 else i * 1e-2 for i in vals_145]
        vals_030 = [i * 1e3 if i < 1 else i * 1e-2 for i in vals_030]
        Tx1 = plt.scatter(x, vals_163, color=colors[0], s=20, label=f'{md_run_numbers[0]}')
        Tx2 = plt.scatter(x, vals_159, color=colors[1], s=20, label=f'{md_run_numbers[1]}')
        Tx3 = plt.scatter(x, vals_145, color=colors[2], s=20, label=f'{md_run_numbers[2]}')
        Tx4 = plt.scatter(x, vals_030, color=colors[3], s=20, label=f'{md_run_numbers[3]}')

        # Tx1 = plt.scatter(x, abs(pos[i][0]), color=colors[0], s=20, label=f'{md_run_numbers[0]}')
        # Tx2 = plt.scatter(x, abs(pos[i][1]), color=colors[1], s=20, label=f'{md_run_numbers[1]}')
        # Tx3 = plt.scatter(x, abs(pos[i][2]), color=colors[2], s=20, label=f'{md_run_numbers[2]}')
        # Tx4 = plt.scatter(x, abs(pos[i][3]), color=colors[3], s=20, label=f'{md_run_numbers[3]}')
        plt.legend(loc='best')
        handles = [mpl_patches.Rectangle((0, 0), 1, 1, fc="white", ec="white",
                lw=0, alpha=0)] * 2
        textstr = '\n'.join((r'LHCb internal', r'magDown'))
        props = dict(boxstyle='square', facecolor='white', alpha=0.7)
        plt.text(0.85, 0.55, textstr, transform=ax.transAxes, fontsize=25,
        verticalalignment='top', bbox=props)
        # labels = []
        # labels.append("LHCb internal")
        # labels.append("magDown")
        # labels.append(f'{md_run_numbers[0]}')
        # ax.legend(handles, labels, loc='best', fontsize=30,
        #         fancybox=False, framealpha=0.7,
        #         handlelength=0,
        #         handletextpad=0)

        plt.tick_params(axis='x', labelrotation = 45, labelsize=16)
        plt.xlabel(r'Module number')

        if i <= 3:
            plt.ylabel(r'T1 modules Rz[mrad]')
        if i > 3 & i <= 7:
            plt.ylabel(r'T2 modules Rz[mrad]')
        if i > 7:
            plt.ylabel(r'T3 modules Rz[mrad]')
        plt.savefig(f"{outdir}/{out_labels[i]}_Rz_constrain.pdf",bbox_inches='tight')
        plt.clf()

    # magup
    # fig1, ax1 = plt.subplots(1,1, figsize=(10,10))
    # r1 = 215/255
    # g1 = 48/255
    # b1 = 39/255
    # r2 = 252/255
    # g2 = 141/255
    # b2 = 89/255
    #
    # pos_up = get_positions(md_vs_mu)
    # # magup vs magdown
    # plt.xticks(x, runs_X2_T3)
    # Tx1 = ax1.scatter(x, pos[3][0], color='black', s=20, label=f'{md_run_numbers[0]}')
    # Tx2 = ax1.scatter(x, pos[3][1], color='red', s=20, label=f'{md_run_numbers[1]}')
    # handles = [mpl_patches.Rectangle((0, 0), 1, 1, fc="white", ec="white",
    #                                 lw=0, alpha=0)] * 2
    # labels = []
    # labels.append("LHCb internal")
    # labels.append(f"magDown")
    #
    # ax1.legend(handles, labels, loc='best', fontsize=30,
    #         fancybox=False, framealpha=0.7,
    #         handlelength=0, handletextpad=0)
    #
    # ax1.tick_params(axis='x', labelrotation = 45, labelsize=16)
    # ax1.set_xlabel(r'Module number')
    # ax1.set_ylabel(r'T3 modules Tx[mm]')
    # plt.savefig(r"T3X2_Tx_md_vs_mu.pdf",bbox_inches='tight')


############## previous coed fragments for testing ###########

# runs_T1_U = []
# runs_T1_V = []
# runs_T1_X1 = []
# runs_T1_X2 = []
#
# runs_T2_U = []
# runs_T2_V = []
# runs_T2_X1 = []
# runs_T2_X2 = []
#
# runs_T3_U = []
# runs_T3_V = []
# runs_T3_X1 = []
# runs_T3_X2 = []
#
# for j in range(0,len(stations)):
#    for k in range(0,len(layers)):
#       if j==0 and k==0:
#          #string = runs[j]
#          runs_T3_U=runs
#       elif j==0 and k==1:
#          for i in range(0,len(runs)):
#             string = runs[i]
#             runs_T3_V.append(string.replace("T3U", "T3V"))
#       elif j==0 and k==2:
#           for i in range(0,len(runs)):
#             string = runs[i]
#             runs_T3_X1.append(string.replace("T3U", "T3X1"))
#       elif j==0 and k==3:
#          for i in range(0,len(runs)):
#             string = runs[i]
#             runs_T3_X2.append(string.replace("T3U", "T3X2"))
#
# # print('runs_T3_U', runs_T3_U)
# # these are used
# T1U_Tx_yml = []
# T1U_Tx = []
# T1V_Tx_yml = []
# T1V_Tx = []
# T1X1_Tx_yml = []
# T1X1_Tx = []
# T1X2_Tx_yml = []
# T1X2_Tx = []
#
# T2U_Tx_yml = []
# T2U_Tx = []
# T2V_Tx_yml = []
# T2V_Tx = []
# T2X1_Tx_yml = []
# T2X1_Tx = []
# T2X2_Tx_yml = []
# T2X2_Tx = []
#
# T3U_Tx_yml = []
# T3U_Tx = []
# T3V_Tx_yml = []
# T3V_Tx = []
# T3X1_Tx_yml = []
# T3X1_Tx = []
# T3X2_Tx_yml = []
# T3X2_Tx = []
#
# for i in range(0,len(runs)):
#    with open(path_run_folder + path_yaml_file, 'r') as stream:
#       data_loaded = yaml.load(stream, Loader=yaml.Loader)
#       '''T1U_Tx_yml.append(data_loaded[runs_T1_U[i]]['position'][0])
#       T1U_Tx.append(float(re.findall(r'\d+',T1U_Tx_yml[i])[0] + "." + re.findall(r'\d+',T1U_Tx_yml[i])[1]))
#
#       T1V_Tx_yml.append(data_loaded[runs_T1_V[i]]['position'][0])
#       T1V_Tx.append(float(re.findall(r'\d+',T1V_Tx_yml[i])[0] + "." + re.findall(r'\d+',T1V_Tx_yml[i])[1]))
#
#       T1X1_Tx_yml.append(data_loaded[runs_T1_X1[i]]['position'][0])
#       T1X1_Tx.append(float(re.findall(r'\d+',T1X1_Tx_yml[i])[0] + "." + re.findall(r'\d+',T1X1_Tx_yml[i])[1]))
#
#       T1X2_Tx_yml.append(data_loaded[runs_T1_X2[i]]['position'][0])
#       T1X2_Tx.append(float(re.findall(r'\d+',T1X2_Tx_yml[i])[0] + "." + re.findall(r'\d+',T1X2_Tx_yml[i])[1]))
#
#       T2U_Tx_yml.append(data_loaded[runs_T2_U[i]]['position'][0])
#       T2U_Tx.append(float(re.findall(r'\d+',T2U_Tx_yml[i])[0] + "." + re.findall(r'\d+',T2U_Tx_yml[i])[1]))
#
#       T2V_Tx_yml.append(data_loaded[runs_T2_V[i]]['position'][0])
#       T2V_Tx.append(float(re.findall(r'\d+',T2V_Tx_yml[i])[0] + "." + re.findall(r'\d+',T2V_Tx_yml[i])[1]))
#
#       T2X1_Tx_yml.append(data_loaded[runs_T2_X1[i]]['position'][0])
#       T2X1_Tx.append(float(re.findall(r'\d+',T2X1_Tx_yml[i])[0] + "." + re.findall(r'\d+',T2X1_Tx_yml[i])[1]))
#
#       T2X2_Tx_yml.append(data_loaded[runs_T2_X2[i]]['position'][0])
#       T2X2_Tx.append(float(re.findall(r'\d+',T2X2_Tx_yml[i])[0] + "." + re.findall(r'\d+',T2X2_Tx_yml[i])[1]))
#       '''
#       T3U_Tx_yml.append(data_loaded[runs_T3_U[i]]['position'][0])
#       T3U_Tx.append(float(re.findall(r'\d+',T3U_Tx_yml[i])[0] + "." + re.findall(r'\d+',T3U_Tx_yml[i])[1]))
#
#       T3V_Tx_yml.append(data_loaded[runs_T3_V[i]]['position'][0])
#       T3V_Tx.append(float(re.findall(r'\d+',T3V_Tx_yml[i])[0] + "." + re.findall(r'\d+',T3V_Tx_yml[i])[1]))
#
#       T3X1_Tx_yml.append(data_loaded[runs_T3_X1[i]]['position'][0])
#       T3X1_Tx.append(float(re.findall(r'\d+',T3X1_Tx_yml[i])[0] + "." + re.findall(r'\d+',T3X1_Tx_yml[i])[1]))
#
#       T3X2_Tx_yml.append(data_loaded[runs_T3_X2[i]]['position'][0])
#       T3X2_Tx.append(float(re.findall(r'\d+',T3X2_Tx_yml[i])[0] + "." + re.findall(r'\d+',T3X2_Tx_yml[i])[1]))
