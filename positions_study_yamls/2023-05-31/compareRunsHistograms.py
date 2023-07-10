import numpy as np
import matplotlib.pyplot as plt
from math import *
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

def diff_to_hist(files, degree_of_freedom, spatial_object):
    '''
        input: list of yaml files
        degrees of freedom: Tx Ty Tz Rx Ry Rz
        spatial_object: position or rotation

        output: return array of per module difference for each layer
        -> 12 outputs or 2D array?
    '''

    # specs
    num_files = len(files)
    iter_num = 0
    DoF_value = 0
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

    runs_T1 = ["T1UHL0Q0M0", "T1UHL0Q0M1", "T1UHL0Q0M2", "T1UHL0Q0M3", "T1UHL0Q0M4", "T1UHL0Q2M0", "T1UHL0Q2M1", "T1UHL0Q2M2", "T1UHL0Q2M3", "T1UHL0Q2M4", "T1UHL1Q1M0", "T1UHL1Q1M1", "T1UHL1Q1M2", "T1UHL1Q1M3", "T1UHL1Q1M4", "T1UHL1Q3M0", "T1UHL1Q3M1", "T1UHL1Q3M2", "T1UHL1Q3M3", "T1UHL1Q3M4"]

    runs_T2 = ["T2UHL0Q0M0", "T2UHL0Q0M1", "T2UHL0Q0M2", "T2UHL0Q0M3", "T2UHL0Q0M4", "T2UHL0Q2M0", "T2UHL0Q2M1", "T2UHL0Q2M2", "T2UHL0Q2M3", "T2UHL0Q2M4", "T2UHL1Q1M0",
            "T2UHL1Q1M1", "T2UHL1Q1M2", "T2UHL1Q1M3", "T2UHL1Q1M4", "T2UHL1Q3M0", "T2UHL1Q3M1", "T2UHL1Q3M2", "T2UHL1Q3M3", "T2UHL1Q3M4"]

    runs = ["T3UHL0Q0M0", "T3UHL0Q0M1", "T3UHL0Q0M2", "T3UHL0Q0M3", "T3UHL0Q0M4", "T3UHL0Q2M0", "T3UHL0Q2M1", "T3UHL0Q2M2", "T3UHL0Q2M3", "T3UHL0Q2M4", "T3UHL1Q1M0"#\
            , "T3UHL1Q1M1", "T3UHL1Q1M2", "T3UHL1Q1M3", "T3UHL1Q1M4", "T3UHL1Q3M0", "T3UHL1Q3M1", "T3UHL1Q3M2", "T3UHL1Q3M3", "T3UHL1Q3M4"]

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
            with open(file, 'r') as stream:
                data_loaded = yaml.load(stream, Loader=yaml.Loader)

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
        # data is filled into the correct lists
    return np.array(T1U_Tx), np.array(T1V_Tx), np.array(T1X1_Tx), np.array(T1X2_Tx), np.array(T2U_Tx), np.array(T2V_Tx), np.array(T2X1_Tx), np.array(T2X2_Tx), np.array(T3U_Tx), np.array(T3V_Tx), np.array(T3X1_Tx), np.array(T3X2_Tx)
    # full_obj = [T1U_Tx, T1V_Tx, T1X1_Tx, T1X2_Tx, T2U_Tx, T2V_Tx, T2X1_Tx, T2X2_Tx, T3U_Tx, T3V_Tx, T3X1_Tx, T3X2_Tx]
    # out_base_to_1 = [[] for _ in range(num_files)]
    # out_base_to_2 = [[] for _ in range(num_files)]
    # out_base_to_3 = [[] for _ in range(num_files)]
    # for obj in full_obj:
    #     out_base_to_1.append(obj[1])
    #     out_base_to_2.append(obj[2])
    #     out_base_to_3.append(obj[3])

    # return np.array(out_base_to_1), np.array(out_base_to_2), np.array(out_base_to_3)


if __name__ == '__main__':
    hep.style.use(hep.style.LHCb2)
    import matplotlib.patches as mpl_patches

    def meta_constructor(loader, node):
       return loader.construct_mapping(node)

    yaml.add_constructor('!alignment', meta_constructor)

    DoF = 'Tx'
    position_or_rotation = 'position'
    MagPol = 'MD'

    stations = ["T1", "T2", "T3"]
    layers = ["U", "V", "X1", "X2"]

    # folder on pc
    path_run_folder = "/mnt/c/Users/Nils/Desktop/Promotion/SciFi/positions_study_yamls/2023-05-31"
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
    ]
    mixed_yaml_files = [
        path_run_folder + '/256290/Modules.yml',
        path_run_folder + '/256267/Modules_run_256267.yml',
        path_run_folder + "/256163/Modules.yml",
        path_run_folder + "/256159/Modules.yml"
    ]
    md_runs = ["/256163/Modules.yml", "/256159/Modules.yml", "/256145/Modules.yml", "/256030/Modules.yml"]
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

    outdir1 = 'MD_outfiles'
    outdir2 = 'MU_vs_MD_outfiles'

    run_labels = [runs_T1, runs_V_T1, runs_X1_T1, runs_X2_T1, runs_T2, runs_V_T2, runs_X1_T2, runs_X2_T2, runs, runs_V_T3, runs_X1_T3, runs_X2_T3]
    out_labels = ['T1U', 'T1V', 'T1X1', 'T1X2', 'T2U', 'T2V', 'T2X1', 'T2X2', 'T3U', 'T3V', 'T3X1', 'T3X2']

    fig, ax = plt.subplots(1,1, figsize=(10,10))
    r1 = 215/255
    g1 = 48/255
    b1 = 39/255
    r2 = 252/255
    g2 = 141/255
    b2 = 89/255

    pos = diff_to_hist(magDown_yaml_files, DoF, position_or_rotation)
    for i in range(12):
        colors = ['black', 'red', 'blue', 'green', 'yellow']

        if DoF == 'Rz':
            runs_md = [[] for _ in range(len(magUp_yaml_files))]
            for j in range(len(runs_md)):
                runs_md[j] = abs(pos[i][j])
                runs_md[j] = [i * 1e3 if i < 1 else i * 1e-2 for i in runs_md[j]]
            base_run = runs_md[0]
            diff1 = base_run - runs_md[1]
            diff2 = base_run - runs_md[2]
            diff3 = base_run - runs_md[3]
            n1, bins1, patches1 = plt.hist(diff1, histtype='step', color=colors[0])
            n2, bins2, patches2 = plt.hist(diff2, histtype='step', color=colors[1])
            n3, bins3, patches3 = plt.hist(diff3, histtype='step', color=colors[2])
        else:
            base_run = abs(pos[i][0])
            diff1 = base_run - abs(pos[i][1])
            diff2 = base_run - abs(pos[i][2])
            diff3 = base_run - abs(pos[i][3])
            n1, bins1, patches1 = plt.hist(diff1, histtype='step', color=colors[0])
            n2, bins2, patches2 = plt.hist(diff2, histtype='step', color=colors[1])
            n3, bins3, patches3 = plt.hist(diff3, histtype='step', color=colors[2])
        # plt.legend(loc='best')
        handles = [mpl_patches.Rectangle((0, 0), 1, 1, fc="white", ec="white",
                lw=0, alpha=0)] * 2
        textstr = '\n'.join((r'LHCb internal', r'magDown'))
        props = dict(boxstyle='square', facecolor='white', alpha=0.7)
        plt.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=25,
        verticalalignment='top', bbox=props)
        plt.xlabel('diff')
        print('############ Debugging bin width ##############')
        print('n1: ', n1)
        print('n2; ', n2)
        print('n3 ', n3)
        print('############# bins ################')
        print('bins1: ', bins1)
        print('bins2: ', bins2)
        print('bins3: ', bins1)
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
        plt.savefig(f"{outdir1}/{DoF}/{out_labels[i]}_{DoF}_{MagPol}_hist_diff.pdf",bbox_inches='tight')
        plt.clf()
    # n2, bins2, patches2 = plt.hist(pos[2])
    # n3, bins3, patches3 = plt.hist(pos[3])
