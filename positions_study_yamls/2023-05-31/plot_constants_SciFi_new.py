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

def get_positions(files, degree_of_freedom, spatial_object):
    '''
      input: - list of files,
             - degree_of_freedom: Tx, Tz, Rz (string)
             - spatial_object: position or rotation (string)

      output: - 12 arrays of Module values for given DoF
    '''

    num_files = len(files)
    iter_num = 0

    dof_value = 0
    if degree_of_freedom == 'Tx':
       dof_value = iter_num
    if degree_of_freedom == 'Tz' or degree_of_freedom == 'Rz':
       dof_value = 2

    PosRot = spatial_object

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

#    print('all files:', files)
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
            #   print(data_loaded[runs_T1_U[iter_num][i]])
              T1U_Tx_yml[iter_num].append(data_loaded[runs_T1_U[iter_num][i]][PosRot][dof_value])
              T1U_Tx[iter_num].append(float(re.findall(r'\d+',T1U_Tx_yml[iter_num][i])[0] + "." + re.findall(r'\d+',T1U_Tx_yml[iter_num][i])[1]))

              T1V_Tx_yml[iter_num].append(data_loaded[runs_T1_V[iter_num][i]][PosRot][dof_value])
              T1V_Tx[iter_num].append(float(re.findall(r'\d+',T1V_Tx_yml[iter_num][i])[0] + "." + re.findall(r'\d+',T1V_Tx_yml[iter_num][i])[1]))

              T1X1_Tx_yml[iter_num].append(data_loaded[runs_T1_X1[iter_num][i]][PosRot][dof_value])
              T1X1_Tx[iter_num].append(float(re.findall(r'\d+',T1X1_Tx_yml[iter_num][i])[0] + "." + re.findall(r'\d+',T1X1_Tx_yml[iter_num][i])[1]))

              T1X2_Tx_yml[iter_num].append(data_loaded[runs_T1_X2[iter_num][i]][PosRot][dof_value])
              T1X2_Tx[iter_num].append(float(re.findall(r'\d+',T1X2_Tx_yml[iter_num][i])[0] + "." + re.findall(r'\d+',T1X2_Tx_yml[iter_num][i])[1]))

              # T2
              T2U_Tx_yml[iter_num].append(data_loaded[runs_T2_U[iter_num][i]][PosRot][dof_value])
              T2U_Tx[iter_num].append(float(re.findall(r'\d+',T2U_Tx_yml[iter_num][i])[0] + "." + re.findall(r'\d+',T2U_Tx_yml[iter_num][i])[1]))

              T2V_Tx_yml[iter_num].append(data_loaded[runs_T2_V[iter_num][i]][PosRot][dof_value])
              T2V_Tx[iter_num].append(float(re.findall(r'\d+',T2V_Tx_yml[iter_num][i])[0] + "." + re.findall(r'\d+',T2V_Tx_yml[iter_num][i])[1]))

              T2X1_Tx_yml[iter_num].append(data_loaded[runs_T2_X1[iter_num][i]][PosRot][dof_value])
              T2X1_Tx[iter_num].append(float(re.findall(r'\d+',T2X1_Tx_yml[iter_num][i])[0] + "." + re.findall(r'\d+',T2X1_Tx_yml[iter_num][i])[1]))

              T2X2_Tx_yml[iter_num].append(data_loaded[runs_T2_X2[iter_num][i]][PosRot][dof_value])
              T2X2_Tx[iter_num].append(float(re.findall(r'\d+',T2X2_Tx_yml[iter_num][i])[0] + "." + re.findall(r'\d+',T2X2_Tx_yml[iter_num][i])[1]))

              # T3
              T3U_Tx_yml[iter_num].append(data_loaded[runs_T3_U[iter_num][i]][PosRot][dof_value])
              T3U_Tx[iter_num].append(float(re.findall(r'\d+',T3U_Tx_yml[iter_num][i])[0] + "." + re.findall(r'\d+',T3U_Tx_yml[iter_num][i])[1]))

              T3V_Tx_yml[iter_num].append(data_loaded[runs_T3_V[iter_num][i]][PosRot][dof_value])
              T3V_Tx[iter_num].append(float(re.findall(r'\d+',T3V_Tx_yml[iter_num][i])[0] + "." + re.findall(r'\d+',T3V_Tx_yml[iter_num][i])[1]))

              T3X1_Tx_yml[iter_num].append(data_loaded[runs_T3_X1[iter_num][i]][PosRot][dof_value])
              T3X1_Tx[iter_num].append(float(re.findall(r'\d+',T3X1_Tx_yml[iter_num][i])[0] + "." + re.findall(r'\d+',T3X1_Tx_yml[iter_num][i])[1]))

              T3X2_Tx_yml[iter_num].append(data_loaded[runs_T3_X2[iter_num][i]][PosRot][dof_value])
              T3X2_Tx[iter_num].append(float(re.findall(r'\d+',T3X2_Tx_yml[iter_num][i])[0] + "." + re.findall(r'\d+',T3X2_Tx_yml[iter_num][i])[1]))
        iter_num += 1
    return np.array(T1U_Tx), np.array(T1V_Tx), np.array(T1X1_Tx), np.array(T1X2_Tx), np.array(T2U_Tx), np.array(T2V_Tx), np.array(T2X1_Tx), np.array(T2X2_Tx), np.array(T3U_Tx), np.array(T3V_Tx), np.array(T3X1_Tx), np.array(T3X2_Tx)

def replace_string(get_len_from_string, arr, str1, str2):
    arr = []
    for i in range(0,len(get_len_from_string)):
       string = get_len_from_string[i]
       arr.append(string.replace(str1, str2))
    return arr


# def multiplot():

#     return

if __name__ == '__main__':
    hep.style.use(hep.style.LHCb2)
    import matplotlib.patches as mpl_patches

    def meta_constructor(loader, node):
       return loader.construct_mapping(node)

    yaml.add_constructor('!alignment', meta_constructor)
    # DoF = 'Tx'
    # position_or_rotation = 'position'

    # DoF = 'Tz'
    # position_or_rotation = 'position'

    DoF = 'Rz'
    position_or_rotation = 'rotation'
    MagPol = 'MD'

    stations = ["T1", "T2", "T3"]
    layers = ["U", "V", "X1", "X2"]

    # folder on pc
    path_run_folder = "/mnt/c/Users/Nils/Desktop/Promotion/SciFi/108th_lhcb_week/positions_study_yamls/2023-05-31"
    # path_run_folder = "/Users/nibreer/Documents/108th_lhcb_week/positions_study_yamls/2023-05-31"
    magUp_yaml_files = [
                        path_run_folder + '/256267/Modules_run_256267.yml',
                        path_run_folder + '/256272/Modules_run_256272.yml',
                        path_run_folder + '/256273/Modules_run_256273.yml',
                        path_run_folder + '/256278/Modules_run_256278.yml',
                        path_run_folder + '/256290/Modules.yml'
                        ]
    magDown_yaml_files = [
                        path_run_folder + "/256145/Modules.yml",
                        path_run_folder + "/256163/Modules.yml",
                        path_run_folder + "/256159/Modules.yml",
                        path_run_folder + "/256030/Modules.yml",
                        path_run_folder + "/255949/Modules_run_255949.yml"
    ]
    mixed_yaml_files = [
        path_run_folder + '/256290/Modules.yml',
        path_run_folder + '/256267/Modules_run_256267.yml',
        path_run_folder + "/256163/Modules.yml",
        path_run_folder + "/256159/Modules.yml"
    ]
    md_runs = ["/256145/Modules.yml", "/256163/Modules.yml", "/256159/Modules.yml", "/256030/Modules.yml", "/255949/Modules_run_255949.yml"]
    mu_runs = ['/256267/Modules_run_256267.yml', '/256272/Modules_run_256272.yml', '/256273/Modules_run_256273.yml', "/256278/Modules_run_256278.yml", '/256290/Modules.yml']
    md_vs_mu = ['/256290/Modules.yml', '/256267/Modules_run_256267.yml', "/256163/Modules.yml", "/256159/Modules.yml"]

    runs_T1 = ["T1UHL0Q0M0", "T1UHL0Q0M1", "T1UHL0Q0M2", "T1UHL0Q0M3", "T1UHL0Q0M4", "T1UHL0Q2M0", "T1UHL0Q2M1", "T1UHL0Q2M2", "T1UHL0Q2M3", "T1UHL0Q2M4", "T1UHL1Q1M0",
    "T1UHL1Q1M1", "T1UHL1Q1M2", "T1UHL1Q1M3", "T1UHL1Q1M4", "T1UHL1Q3M0", "T1UHL1Q3M1", "T1UHL1Q3M2", "T1UHL1Q3M3", "T1UHL1Q3M4"]

    runs_T2 = ["T2UHL0Q0M0", "T2UHL0Q0M1", "T2UHL0Q0M2", "T2UHL0Q0M3", "T2UHL0Q0M4", "T2UHL0Q2M0", "T2UHL0Q2M1", "T2UHL0Q2M2", "T2UHL0Q2M3", "T2UHL0Q2M4", "T2UHL1Q1M0",
    "T2UHL1Q1M1", "T2UHL1Q1M2", "T2UHL1Q1M3", "T2UHL1Q1M4", "T2UHL1Q3M0", "T2UHL1Q3M1", "T2UHL1Q3M2", "T2UHL1Q3M3", "T2UHL1Q3M4"]

    runs = ["T3UHL0Q0M0", "T3UHL0Q0M1", "T3UHL0Q0M2", "T3UHL0Q0M3", "T3UHL0Q0M4", "T3UHL0Q2M0", "T3UHL0Q2M1", "T3UHL0Q2M2", "T3UHL0Q2M3", "T3UHL0Q2M4", "T3UHL1Q1M0"#\
    , "T3UHL1Q1M1", "T3UHL1Q1M2", "T3UHL1Q1M3", "T3UHL1Q1M4", "T3UHL1Q3M0", "T3UHL1Q3M1", "T3UHL1Q3M2", "T3UHL1Q3M3", "T3UHL1Q3M4"]
    # list of object names for all layers

    # T1
    runs_V_T1 = []
    runs_V_T1 = replace_string(runs_T1, runs_V_T1, "U", "V")

    runs_X1_T1 = []
    runs_X1_T1 = replace_string(runs_T1, runs_X1_T1, "U", "X1")

    runs_X2_T1 = []
    runs_X2_T1 = replace_string(runs_T1, runs_X2_T1, "U", "X2")

    # T2
    runs_V_T2 = []
    runs_V_T2 = replace_string(runs_T2, runs_V_T2, "U", "V")

    runs_X1_T2 = []
    runs_X1_T2 = replace_string(runs_T2, runs_X1_T2, "U", "X1")

    runs_X2_T2 = []
    runs_X2_T2 = replace_string(runs_T2, runs_X2_T2, "U", "X2")

    # T3
    runs_V_T3 = []
    runs_V_T3 = replace_string(runs, runs_V_T3, "U", "V")

    runs_X1_T3 = []
    runs_X1_T3 = replace_string(runs, runs_X1_T3, "U", "X1")

    runs_X2_T3 = []
    runs_X2_T3 = replace_string(runs, runs_X2_T3, "U", "X2")

    x = list(range(len(runs)))

    # extract run numbers from file names
    mu_run_numbers, md_run_numbers, mixed_run_numbers = [], [], []
    for file in md_runs:
        num = re.findall(r'\d+', file)
        md_run_numbers.append(num[0])
    for file in mu_runs:
        num = re.findall(r'\d+', file)
        mu_run_numbers.append(num[0])
    for file in md_vs_mu:
        num = re.findall(r'\d+', file)
        mixed_run_numbers.append(num[0])

    outdir1 = 'MD_outfiles'
    outdir2 = 'MU_vs_MD_outfiles'

    pos = get_positions(magUp_yaml_files, DoF, position_or_rotation)

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

    # magnet down run order
    run_order = ['256145','256163','256159','256030']

    # magUp run_order
    mu_run_order = ['256267', '267272', '256273', '256278', '256290']

    for i in range(12):
        plt.xticks(x, run_labels[i])
        colors = ['black', 'red', 'blue', 'green', 'yellow']

        if DoF == 'Rz':
            runs_md = [[] for _ in range(len(magUp_yaml_files))]
            for j in range(len(runs_md)):
                runs_md[j] = abs(pos[i][j])
                # print('BEFORE: ', runs_md[j])
                runs_md[j] = [i * 1e3 if i < 1 else i * 1e-2 for i in runs_md[j]]
                # print('AFTER: ', runs_md[j])
            base_run = runs_md[0]
            Tx1 = plt.scatter(x, runs_md[0], color=colors[0], s=20, label=f'{md_run_numbers[0]}')
            Tx2 = plt.scatter(x, runs_md[1], color=colors[1], s=20, label=f'{md_run_numbers[1]}')
            Tx3 = plt.scatter(x, runs_md[2], color=colors[2], s=20, label=f'{md_run_numbers[2]}')
            Tx4 = plt.scatter(x, runs_md[3], color=colors[3], s=20, label=f'{md_run_numbers[3]}')
            Tx5 = plt.scatter(x, runs_md[4], color=colors[4], s=20, label=f'{md_run_numbers[4]}')
        else:
            Tx1 = plt.scatter(x, abs(pos[i][0]), color=colors[0], s=20, label=f'{md_run_numbers[0]}')
            Tx2 = plt.scatter(x, abs(pos[i][1]), color=colors[1], s=20, label=f'{md_run_numbers[1]}')
            Tx3 = plt.scatter(x, abs(pos[i][2]), color=colors[2], s=20, label=f'{md_run_numbers[2]}')
            Tx4 = plt.scatter(x, abs(pos[i][3]), color=colors[3], s=20, label=f'{md_run_numbers[3]}')
            Tx5 = plt.scatter(x, abs(pos[i][4]), color=colors[4], s=20, label=f'{md_run_numbers[4]}')
        plt.legend(loc='best')
        handles = [mpl_patches.Rectangle((0, 0), 1, 1, fc="white", ec="white",
                lw=0, alpha=0)] * 2
        textstr = '\n'.join((r'LHCb internal', r'magDown'))
        props = dict(boxstyle='square', facecolor='white', alpha=0.7)
        plt.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=25,
        verticalalignment='top', bbox=props)

        plt.tick_params(axis='x', labelrotation = 45, labelsize=16)
        plt.xlabel(r'Module number')

        if position_or_rotation == 'position':
            if i <= 3:
                plt.ylabel(f'T1 modules {DoF}[mm]')
            if i > 3 & i <= 7:
                plt.ylabel(f'T2 modules {DoF}[mm]')
            if i > 7:
                plt.ylabel(f'T3 modules {DoF}[mm]')
        else:
            if i <= 3:
                plt.ylabel(f'T1 modules {DoF}[mrad]')
            if i > 3 & i <= 7:
                plt.ylabel(f'T2 modules {DoF}[mrad]')
            if i > 7:
                plt.ylabel(f'T3 modules {DoF}[mrad]')
        plt.savefig(f"{outdir1}/{DoF}/{out_labels[i]}_{DoF}_{MagPol}_constrain.pdf",bbox_inches='tight')
        plt.clf()

    # magup
    fig1, ax1 = plt.subplots(1,1, figsize=(10,10))
    r1 = 215/255
    g1 = 48/255
    b1 = 39/255
    r2 = 252/255
    g2 = 141/255
    b2 = 89/255

    # print('md run numbers:', mu_run_numbers)
    # print('mixed run numbers:', mixed_run_numbers)

    pos_up = get_positions(mixed_yaml_files, DoF, position_or_rotation)
    # magup vs magdown
    for i in range(12):
        plt.xticks(x, run_labels[i])
        colors = ['black', 'red', 'blue', 'green']

        if DoF == 'Rz':
            runs = [[] for _ in range(len(md_vs_mu))]
            for j in range(len(runs)):
                runs[j] = abs(pos_up[i][j])
                runs[j] = [i * 1e3 if i < 1 else i * 1e-2 for i in runs[j]]
            Tx1 = plt.scatter(x, runs[0], color=colors[0], s=20, label=f'{mixed_run_numbers[0]}')
            Tx2 = plt.scatter(x, runs[1], color=colors[1], s=20, label=f'{mixed_run_numbers[1]}')
            Tx3 = plt.scatter(x, runs[2], color=colors[2], s=20, label=f'{mixed_run_numbers[2]}')
            Tx4 = plt.scatter(x, runs[3], color=colors[3], s=20, label=f'{mixed_run_numbers[3]}')
        else:
            Tx1 = plt.scatter(x, abs(pos_up[i][0]), color=colors[0], s=20, label=f'{mixed_run_numbers[0]}')
            Tx2 = plt.scatter(x, abs(pos_up[i][1]), color=colors[1], s=20, label=f'{mixed_run_numbers[1]}')
            Tx3 = plt.scatter(x, abs(pos_up[i][2]), color=colors[2], s=20, label=f'{mixed_run_numbers[2]}')
            Tx4 = plt.scatter(x, abs(pos_up[i][3]), color=colors[3], s=20, label=f'{mixed_run_numbers[3]}')
        plt.legend(loc='best')
        handles = [mpl_patches.Rectangle((0, 0), 1, 1, fc="white", ec="white",
                lw=0, alpha=0)] * 2
        textstr = '\n'.join((r'LHCb internal', r'MU vs. MD'))
        props = dict(boxstyle='square', facecolor='white', alpha=0.7)
        plt.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=20,
        verticalalignment='top', bbox=props)

        plt.tick_params(axis='x', labelrotation = 45, labelsize=16)
        plt.xlabel(r'Module number')
        if position_or_rotation == 'position':
            if i <= 3:
                plt.ylabel(f'T1 modules {DoF}[mm]')
            if i > 3 & i <= 7:
                plt.ylabel(f'T2 modules {DoF}[mm]')
            if i > 7:
                plt.ylabel(f'T3 modules {DoF}[mm]')
        else:
            if i <= 3:
                plt.ylabel(f'T1 modules {DoF}[mrad]')
            if i > 3 & i <= 7:
                plt.ylabel(f'T2 modules {DoF}[mrad]')
            if i > 7:
                plt.ylabel(f'T3 modules {DoF}[mrad]')
        plt.savefig(f"{outdir2}/{DoF}/{out_labels[i]}_{DoF}_MU_vs_MD.pdf",bbox_inches='tight')
        plt.clf()
