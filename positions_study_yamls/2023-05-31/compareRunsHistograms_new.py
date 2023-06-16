import numpy as np
import matplotlib.pyplot as plt
from math import *
import ROOT
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

labels = ['Tx', 'Ty', 'Tz', 'Rx', 'Ry', 'Rz']
positions = ['x_global', 'y_global', 'z_global']
trackInfo = ['nTracks', 'nHits']

regex_typelabel=re.compile("Q")
regex_amodule=re.compile("dPosXYZ")
regex_rmodule=re.compile("dRotXYZ")

def total_layer_diff(arr):
    arr = abs(arr)
    return np.sum(arr)

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
    if degree_of_freedom == 'Tx' or degree_of_freedom == 'Rx':
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
        # the output:
        # [0, 11]: information about the module positions
        # [12, 13, 14]: x ticks, y ticks, z ticks
        # [15, 16, 17]: T1Tx, T1Ty, T1Tz data
        # [18, 19, 20]: T2Tx, T2Ty, T2Tz data
        # [21, 22, 23]: T3Tx, T3Ty, T3Tz data

        # does not work like that, need to wrte some redundant code as in biljanas script :(
    ticksT3 = [runs_T3_U, runs_T3_V,runs_T3_X1, runs_T3_X2]
    ypointsT3Tx = [T3U_Tx, T3V_Tx, T3X1_Tx, T3X2_Tx]
    ypointsT3Tz = [T3U_Tz, T3V_Tz, T3X1_Tz, T3X2_Tz]
    ypointsT3Rz = [T3U_Rz, T3V_Rz, T3X1_Rz, T3X2_Rz]

    ticksT2 = [runs_T2_U, runs_T2_V,runs_T2_X1, runs_T2_X2]
    ypointsT2Tx = [T2U_Tx, T2V_Tx, T2X1_Tx, T2X2_Tx]
    ypointsT2Tz = [T2U_Tz, T2V_Tz, T2X1_Tz, T2X2_Tz]
    ypointsT2Rz = [T2U_Rz, T2V_Rz, T2X1_Rz, T2X2_Rz]

    ticksT1 = [runs_T1_U, runs_T1_V,runs_T1_X1, runs_T1_X2]
    ypointsT1Tz = [T1U_Tz, T1V_Tz, T1X1_Tz, T1X2_Tz]
    ypointsT1Tx = [T1U_Tx, T1V_Tx, T1X1_Tx, T1X2_Tx]
    ypointsT1Rz = [T1U_Rz, T1V_Rz, T1X1_Rz, T1X2_Rz]

    return np.array(T1U_Tx), np.array(T1V_Tx), np.array(T1X1_Tx), np.array(T1X2_Tx), np.array(T2U_Tx), np.array(T2V_Tx), np.array(T2X1_Tx), np.array(T2X2_Tx), np.array(T3U_Tx), np.array(T3V_Tx), np.array(T3X1_Tx), np.array(T3X2_Tx), ticksT1, ticksT2, TicksT3, ypointsT1Tx, ypointsT1Ty, ypointsT1Tz, ypointsT2Tx, ypointsT2Ty, ypointsT2Tz, ypointsT3Tx, ypointsT3Ty, ypointsT3Tz

def plot_modules(arr, DoF, labels, MagPol, outdir, spatial_degree, run_nums):
    layer_displacement_abs_Tx = np.empty([12, 4])
    layer_displacement_abs_Tz = np.empty([12, 4])
    layer_displacement_abs_Rx = np.empty([12, 4])
    layer_displacement_abs_Rz = np.empty([12, 4])
    
    positions_of_modules = diff_to_hist(arr, DoF, spatial_degree)
    for i in range(12):
        # make the limits by eye for now
        Bins_all = np.linspace(-0.4, 0.4, 8)
        Bins_abs = np.linspace(0, 0.4, 8)

        colors = ['black', 'red', 'blue', 'green', 'magenta']

        if DoF == 'Rz' or DoF == 'Rx':
            runs_md = [[] for _ in range(len(arr))]
            for j in range(len(runs_md)):
                runs_md[j] = positions_of_modules[i][j]
                runs_md[j] = [i * 1e3 if i < 1 else i * 1e-2 for i in runs_md[j]]
            base_run = runs_md[0]

            diff1 = np.array(base_run) - np.array(runs_md[1])
            diff2 = np.array(base_run) - np.array(runs_md[2])
            diff3 = np.array(base_run) - np.array(runs_md[3])
            diff4 = np.array(base_run) - np.array(runs_md[4])

            diff_all = np.array([diff1, diff2, diff3, diff4])
            diff_all = diff_all.reshape(4, 20).T
            diff_abs = np.array([abs(diff1), abs(diff2), abs(diff3), abs(diff4)])
            diff_abs = diff_abs.reshape(4, 20).T

            d1 = total_layer_diff(diff1)
            d2 = total_layer_diff(diff2)
            d3 = total_layer_diff(diff3)
            d4 = total_layer_diff(diff4)

            layer_displacement_abs_Rz[i][0] = float("{:.4f}".format(total_layer_diff(diff1)))
            layer_displacement_abs_Rz[i][1] = float("{:.4f}".format(total_layer_diff(diff2)))
            layer_displacement_abs_Rz[i][2] = float("{:.4f}".format(total_layer_diff(diff3)))
            layer_displacement_abs_Rz[i][3] = float("{:.4f}".format(total_layer_diff(diff4)))

            fig, ((ax0, ax1)) = plt.subplots(nrows=2, ncols=1)

            # ax0.hist(diffs, bins=Bins, histtype='bar', label=md_run_numbers[1:])
            ax0.hist(diff_all, bins=Bins_all, density=True, histtype='bar', stacked=True, label=run_nums[1:])
            ax1.hist(diff_abs, bins=Bins_abs, density=True, histtype='bar', stacked=True, label=run_nums[1:])

            ax0.legend(loc='best')
            ax1.legend(loc='best')

            ax0.set_title(f'diff to run 256145, {labels[i]}')
            ax1.set_title(f'absolute diff to run 256145, {labels[i]}')

            fig.tight_layout()
            fig.savefig(f"{outdir}/{DoF}/{labels[i]}_{DoF}_{MagPol}_hist_diff.pdf",bbox_inches='tight')
        else:
            base_run = abs(positions_of_modules[i][0])

            diff1 = base_run - positions_of_modules[i][1]
            diff2 = base_run - positions_of_modules[i][2]
            diff3 = base_run - positions_of_modules[i][3]
            diff4 = base_run - positions_of_modules[i][4]

            diff_all = np.array([diff1, diff2, diff3, diff4])
            diff_all = diff_all.reshape(4, 20).T
            diff_abs = np.array([abs(diff1), abs(diff2), abs(diff3), abs(diff4)])
            diff_abs = diff_abs.reshape(4, 20).T

            d1 = total_layer_diff(diff1)
            d2 = total_layer_diff(diff2)
            d3 = total_layer_diff(diff3)
            d4 = total_layer_diff(diff4)

            if DoF == 'Tx':
                layer_displacement_abs_Tx[i][0] = float("{:.4f}".format(total_layer_diff(diff1)))
                layer_displacement_abs_Tx[i][1] = float("{:.4f}".format(total_layer_diff(diff2)))
                layer_displacement_abs_Tx[i][2] = float("{:.4f}".format(total_layer_diff(diff3)))
                layer_displacement_abs_Tx[i][3] = float("{:.4f}".format(total_layer_diff(diff4)))
            if DoF == 'Tz':
                layer_displacement_abs_Tz[i][0] = float("{:.4f}".format(total_layer_diff(diff1)))
                layer_displacement_abs_Tz[i][1] = float("{:.4f}".format(total_layer_diff(diff2)))
                layer_displacement_abs_Tz[i][2] = float("{:.4f}".format(total_layer_diff(diff3)))
                layer_displacement_abs_Tz[i][3] = float("{:.4f}".format(total_layer_diff(diff4)))
            fig, ((ax0, ax1)) = plt.subplots(nrows=2, ncols=1)
            # ax0.hist(diffs, bins=Bins, histtype='bar', label=md_run_numbers[1:])
            ax0.hist(diff_all, bins=Bins_all, density=True, histtype='bar', stacked=True, label=run_nums[1:])
            ax1.hist(diff_abs, bins=Bins_abs, density=True, histtype='bar', stacked=True, label=run_nums[1:])
            ax0.legend(loc='best')
            ax1.legend(loc='best')
            ax0.set_title(f'diff to run 256145, {out_labels[i]}')
            ax1.set_title(f'absolute diff to run 256145, {out_labels[i]}')
            fig.tight_layout()
            fig.savefig(f"{outdir}/{DoF}/{out_labels[i]}_{DoF}_{MagPol}_hist_diff.pdf",bbox_inches='tight')
    if DoF == 'Tx':
        print('###### Tx ######')
        print('      256163 256159 256030 255949')
        for row in range(12):
            if "U" in out_labels[row] or "V" in out_labels[row]:
                print(f'{out_labels[row]}  ', layer_displacement_abs_Tx[row])
            else:
                print(f'{out_labels[row]} ', layer_displacement_abs_Tx[row])
    if DoF == 'Tz':
        print('###### Tz ######')
        print('      256163 256159 256030 255949')
        for row in range(12):
            print(f'{out_labels[row]} ', layer_displacement_abs_Tz[row])
    if DoF == 'Rz':
        print('###### Rz ######')
        print('      256163 256159 256030 255949')
        for row in range(12):
            print(f'{out_labels[row]} ', layer_displacement_abs_Rz[row])
    if DoF == 'Rx':
        print('###### Rx ######')
        print('      256163 256159 256030 255949')
        for row in range(12):
            print(f'{out_labels[row]} ', layer_displacement_abs_Rx[row])

def open_alignment(thisfile, convergence=True):
    with open(thisfile) as f:
        align_output = json.load(f)

    convergences = align_output.pop('converged')

    for alignable in align_output.keys():
        for label in labels + positions + trackInfo:
            if 'FT' in alignable:
                align_output[alignable][label] = [float(ele.strip(',')) for ele in align_output[alignable][label]]

    if convergence:
        align_output['convergence'] = convergences
    return align_output

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

def plotTxTzMapsGlobal(align_output,stationIn=["T1"],quarters=[0,1],maxmodule=5,index=0,color="C0",txsep=3,tzsep=10):
    
    for jj,station in enumerate(stationIn):
        tzlocal=jj*tzsep*4
        for ii,layer in enumerate(["X1","U","V","X2"]):
            tzlocal=ii*tzsep
            for quarter in quarters:
                for module in range(0,maxmodule):
                    txlocal=(module+1)*txsep if quarter%2==0 else -(module+1)*txsep
                    halflayer=quarter%2
                    plt.scatter(align_output[station+layer+f"HL{halflayer}/Q{quarter}M{module}"]["Tx"][index]+txlocal,
                                align_output[station+layer+f"HL{halflayer}/Q{quarter}M{module}"]["Tz"][index]+tzlocal,
                                color=color)

if __name__ == '__main__':
    hep.style.use(hep.style.LHCb2)
    import matplotlib.patches as mpl_patches

    def meta_constructor(loader, node):
       return loader.construct_mapping(node)

    yaml.add_constructor('!alignment', meta_constructor)

    magnet_polarity = 'MD'

    stations = ["T1", "T2", "T3"]
    layers = ["U", "V", "X1", "X2"]

    # folder on pc
    # path_run_folder = "/mnt/c/Users/Nils/Desktop/Promotion/SciFi/108th_lhcb_week/positions_study_yamls/2023-05-31"
    path_run_folder = "/Users/nibreer/Documents/108th_lhcb_week/positions_study_yamls/2023-05-31"

    # input files lists of various sorts
    ## magDown
    md_runs = [
        "/256145/Modules.yml", 
        "/256163/Modules.yml", 
        "/256159/Modules.yml", 
        "/256030/Modules.yml", 
        "/255949/Modules_run_255949.yml"
    ]
    # magUp
    mu_runs = [
        '/256267/Modules_run_256267.yml', 
        '/256272/Modules_run_256272.yml', 
        '/256273/Modules_run_256273.yml', 
        "/256278/Modules_run_256278.yml", 
        '/256290/Modules.yml'
    ]
    # mixed
    # md_vs_mu = [
    #     '/256290/Modules.yml', 
    #     '/256267/Modules_run_256267.yml', 
    #     "/256163/Modules.yml", 
    #     "/256159/Modules.yml"
    # ]

    magUp_yaml_files = [
                        path_run_folder + mu_runs[0],
                        path_run_folder + mu_runs[1],
                        path_run_folder + mu_runs[2],
                        path_run_folder + mu_runs[3],
                        path_run_folder + mu_runs[4]
                        ]
    magDown_yaml_files = [
                        path_run_folder + md_runs[0],
                        path_run_folder + md_runs[1],
                        path_run_folder + md_runs[2],
                        path_run_folder + md_runs[3],
                        path_run_folder + md_runs[4]
    ]
    # mixed_yaml_files = [
    #     path_run_folder + '/256290/Modules.yml',
    #     path_run_folder + '/256267/Modules_run_256267.yml',
    #     path_run_folder + "/256163/Modules.yml",
    #     path_run_folder + "/256159/Modules.yml"
    # ]

    runs_T1 = ["T1UHL0Q0M0", "T1UHL0Q0M1", "T1UHL0Q0M2", "T1UHL0Q0M3", "T1UHL0Q0M4", "T1UHL0Q2M0", "T1UHL0Q2M1", "T1UHL0Q2M2", "T1UHL0Q2M3", "T1UHL0Q2M4", "T1UHL1Q1M0", "T1UHL1Q1M1", "T1UHL1Q1M2", "T1UHL1Q1M3", "T1UHL1Q1M4", "T1UHL1Q3M0", "T1UHL1Q3M1", "T1UHL1Q3M2", "T1UHL1Q3M3", "T1UHL1Q3M4"]

    runs_T2 = ["T2UHL0Q0M0", "T2UHL0Q0M1", "T2UHL0Q0M2", "T2UHL0Q0M3", "T2UHL0Q0M4", "T2UHL0Q2M0", "T2UHL0Q2M1", "T2UHL0Q2M2", "T2UHL0Q2M3", "T2UHL0Q2M4", "T2UHL1Q1M0",
            "T2UHL1Q1M1", "T2UHL1Q1M2", "T2UHL1Q1M3", "T2UHL1Q1M4", "T2UHL1Q3M0", "T2UHL1Q3M1", "T2UHL1Q3M2", "T2UHL1Q3M3", "T2UHL1Q3M4"]

    runs = ["T3UHL0Q0M0", "T3UHL0Q0M1", "T3UHL0Q0M2", "T3UHL0Q0M3", "T3UHL0Q0M4", "T3UHL0Q2M0", "T3UHL0Q2M1", "T3UHL0Q2M2", "T3UHL0Q2M3", "T3UHL0Q2M4", "T3UHL1Q1M0"#\
            , "T3UHL1Q1M1", "T3UHL1Q1M2", "T3UHL1Q1M3", "T3UHL1Q1M4", "T3UHL1Q3M0", "T3UHL1Q3M1", "T3UHL1Q3M2", "T3UHL1Q3M3", "T3UHL1Q3M4"]

    # list of object names for all layers

    x = list(range(len(runs)))

    # extract run numbers from file names
    mu_run_numbers, md_run_numbers = [], []
    for file in md_runs:
        num = re.findall(r'\d+', file)
        md_run_numbers.append(num[0])
    for file in mu_runs:
        num = re.findall(r'\d+', file)
        mu_run_numbers.append(num[0])

    # run_labels = [runs_T1, runs_V_T1, runs_X1_T1, runs_X2_T1, runs_T2, runs_V_T2, runs_X1_T2, runs_X2_T2, runs, runs_V_T3, runs_X1_T3, runs_X2_T3]
    out_labels = ['T1U', 'T1V', 'T1X1', 'T1X2', 'T2U', 'T2V', 'T2X1', 'T2X2', 'T3U', 'T3V', 'T3X1', 'T3X2']

    # use the run_labels for x ticks
    # but split between top and bottom half modules

    ### output directories depending on what to plot
    hist_outdir = 'hist_out'
    outdir1 = 'MD_outfiles'
    outdir2 = 'MU_vs_MD_outfiles'

    plotted_modules = plot_modules(magDown_yaml_files, 'Tx', out_labels, magnet_polarity, hist_outdir, 'position', md_run_numbers)
    # plt.figure()
    # plotted_modules = plot_modules(magDown_yaml_files, 'Tz', out_labels, magnet_polarity, hist_outdir, 'position', md_run_numbers)
    # plt.figure()
    # plotted_modules = plot_modules(magDown_yaml_files, 'Rz', out_labels, magnet_polarity, hist_outdir, 'rotation', md_run_numbers)

    # module positions inside the plot
    T1_T2_modules = [int(x) for x in np.linspace(-5, 5, 11) if x != 0]
    T3_modules = [int(x) for x in np.linspace(-6, -6, 13) if x != 0]

    fig, ax = plt.subplots(2,2)

    sum = -1
    for i in range(0,2):
        for j in range(0,2):
            sum = sum + 1
            plt.sca(ax[i,j])
            plt.xticks(x, ticksT3[sum]) #here
            ax[i,j].scatter(x, ypointsT3Rz[sum], color='red', s=20) ##here
            ax[i,j].tick_params(axis='x', labelrotation = 90, labelsize=12)
            ax[i,j].set_xlabel(r'Module number')
            ax[i,j].set_ylabel(r'T3 modules Rz[mrad]') ##here
            ax[i,j].text(0.1, 0.7, layers[sum], transform=ax[i,j].transAxes, weight="bold")

    plt.savefig("out_biljana.pdf",bbox_inches='tight') #here
    plt.close()

    ######## 1.) new plots, plotting arbitrary x range e.g. np.linspace vs global Module y values -> so that i can see what changed between runs
    ######## 2.) in the same style plot difference to survey or run 256145 (diff as in previous but with positions)
    ######## 3.) plot global z position of stations vs global Tx
    ######## -> for Tx plot bothe the highest and lowwest Tx and connect it with a line so it looks like a layer
    # layour for 1:
    #   C-side top      A-side top
    #       Q2              Q3
    #             .  |  .    .   .   .    
    #            .   |              .       something like that may come out (or not) 
    #          .     |            .         y axis: global y in this case
    #        .       |           .          x axis: module positions (-5, -4, -3, -2, -1, 1, 2, 3, 4, 5) 10 modules top and 10 bottom
    #      .         |         .
    # ---------------+-----------------
    #    .           |      .
    #   .            |    .
    #  .             |   .
    # .   .   .    . |  .
    #                |
    # C-side bottom     A-side bottom
    #       Q0             Q1