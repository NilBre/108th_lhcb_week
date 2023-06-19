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

regex_typelabel=re.compile("Q")
regex_amodule=re.compile("dPosXYZ")
regex_rmodule=re.compile("dRotXYZ")
labels=["Tx","Ty","Tz","Rx","Ry","Rz"]
positions=["x_global","y_global","z_global"]
trackInfo=["nTracks","nHits"]
stations = ["T1", "T2", "T3"]
layers = ["U", "V", "X1", "X2"]

outname_prefix = 'SciFiAlignv3/'

def plot(data_arr, outname, run_labels, layer_names):
    total_layer_num = len(layer_names)
    total_num_runs = len(run_labels)
    print(total_num_runs)
    x = np.linspace(0, 5, 5)
    colors = ['black', 'blue', 'red', 'green', 'yellow', 'magenta']
    markers = ['o', 'x', 'd', 'D', '.']
    for i in range(total_num_runs):
        print(colors[i])
        x1 = data_arr[i][0:5] # Q0
        x2 = data_arr[i][5:10] # Q2
        x3 = data_arr[i][10:15] # Q1
        x4 = data_arr[i][15:20] # Q3
        #
        # glob1 = glob_data[i][0:5] # Q0
        # glob2 = glob_data[i][5:10] # Q2
        # glob3 = glob_data[i][10:15] # Q1
        # glob4 = glob_data[i][15:20] # Q3
        #
        # shifted_positions1 = np.array(glob1) + np.array(x1)
        # shifted_positions2 = np.array(glob2) + np.array(x2)
        # shifted_positions3 = np.array(glob3) + np.array(x3)
        # shifted_positions4 = np.array(glob4) + np.array(x4)

        ax = [plt.subplot(2,2,i+1) for i in range(4)]
        plt.figure()
        count = 0
        for a in ax:
            plt.sca(a)
            if count == 0: # Q2
                plt.scatter(x, x2[::-1], color=colors[i], marker=markers[i], s=10)
                plt.ylabel('local position [mm]')
                plt.title(f'{outname} local positions')
                a.invert_yaxis()
            if count == 1: # Q3
                plt.scatter(x, x4, color=colors[i], marker=markers[i], s=10, label = f'{run_labels[i]}')
                plt.title(f'layer {layer_names[0]}')
                plt.legend(loc='best')                
            if count == 2: # Q0
                plt.scatter(x, x1[::-1], color=colors[i], marker=markers[i], s=10)
                plt.xticks(x, ["T3UHL0Q0M4", "T3UHL0Q0M3", "T3UHL0Q0M2", "T3UHL0Q0M1", "T3UHL0Q0M0"], rotation=45, fontsize=5)
            if count == 3: # Q1
                plt.scatter(x, x3, color=colors[i], marker=markers[i], s=10)
                a.invert_yaxis()
                plt.xticks(x, ["T3UHL0Q0M0", "T3UHL0Q0M1", "T3UHL0Q0M2", "T3UHL0Q0M3", "T3UHL0Q0M4"], rotation=45, fontsize=5)
            count += 1
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.savefig(f'{outname_prefix}' + run_labels[i] + '_' + outname + '.pdf')

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

    #fix floats
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
    T1X1_PosRot_yml = [[] for _ in range(num_files)]
    T1V_PosRot = [[] for _ in range(num_files)]
    T1X1_PosRot = [[] for _ in range(num_files)]
    T1X2_PosRot_yml = [[] for _ in range(num_files)]
    T1X2_PosRot = [[] for _ in range(num_files)]

    T2U_PosRot_yml = [[] for _ in range(num_files)]
    T2U_PosRot = [[] for _ in range(num_files)]
    T2V_PosRot_yml = [[] for _ in range(num_files)]
    T2X1_PosRot_yml = [[] for _ in range(num_files)]
    T2V_PosRot = [[] for _ in range(num_files)]
    T2X1_PosRot = [[] for _ in range(num_files)]
    T2X2_PosRot_yml = [[] for _ in range(num_files)]
    T2X2_PosRot = [[] for _ in range(num_files)]

    T3U_PosRot_yml = [[] for _ in range(num_files)]
    T3U_PosRot = [[] for _ in range(num_files)]
    T3V_PosRot_yml = [[] for _ in range(num_files)]
    T3X1_PosRot_yml = [[] for _ in range(num_files)]
    T3V_PosRot = [[] for _ in range(num_files)]
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

    # runs = ["T3UHL0Q0M0", "T3UHL0Q0M1", "T3UHL0Q0M2", "T3UHL0Q0M3", "T3UHL0Q0M4", "T3UHL0Q2M0", "T3UHL0Q2M1", "T3UHL0Q2M2", "T3UHL0Q2M3", "T3UHL0Q2M4", "T3UHL1Q1M0"#\
    #         , "T3UHL1Q1M1", "T3UHL1Q1M2", "T3UHL1Q1M3", "T3UHL1Q1M4", "T3UHL1Q3M0", "T3UHL1Q3M1", "T3UHL1Q3M2", "T3UHL1Q3M3", "T3UHL1Q3M4"]

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
            # print(i)
            with open(file, 'r') as stream:
                data_loaded = align_output[iter_num]
                # print(data_loaded)

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

def plotTxTzMapsGlobal(align_output, files, run_labels, layer_names):
    tx = get_data(files, 'Tx', align_output)
    ty = get_data(files, 'Ty', align_output)
    x_glob = get_data(files, 'x_global', align_output)
    y_glob = get_data(files, 'y_global', align_output)
    print(x_glob)
    # for U layer here
    tx_data = tx[0]
    ty_data = ty[0]
    x_g = x_glob[0]
    y_g = y_glob[0]
    plot(tx_data, 'tx_all_runs', run_labels, layer_names)
    plot(ty_data, 'ty_all_runs', run_labels, layer_names)
    plot(x_g, 'x_global', run_labels, layer_names)

files = [\
         "align_logfiles_stability/json_files/parsedlog_256145.json",
         "align_logfiles_stability/json_files/parsedlog_256163.json",
         "align_logfiles_stability/json_files/parsedlog_256159.json",
         "align_logfiles_stability/json_files/parsedlog_256030.json",
]

legendlabels=[\
              "256145",
              "256163",
              "256159",
              "256030",
]

layers = ['T1U', 'T1V', 'T1X1', 'T1X2', 'T2U', 'T2V', 'T2X1', 'T2X2', 'T3U', 'T3V', 'T3X1', 'T3X2']

runs = ["T3UHL0Q0M0", "T3UHL0Q0M1", "T3UHL0Q0M2", "T3UHL0Q0M3", "T3UHL0Q0M4", "T3UHL0Q2M0", "T3UHL0Q2M1", "T3UHL0Q2M2", "T3UHL0Q2M3", "T3UHL0Q2M4", "T3UHL1Q1M0"#\
        , "T3UHL1Q1M1", "T3UHL1Q1M2", "T3UHL1Q1M3", "T3UHL1Q1M4", "T3UHL1Q3M0", "T3UHL1Q3M1", "T3UHL1Q3M2", "T3UHL1Q3M3", "T3UHL1Q3M4"]

align_outputs=[open_alignment(thisfile) for thisfile in files]
plotted_alignables=[]
for align_block in align_outputs:
    thislist=[]
    for key in align_block.keys():
        if "FT" in key:
            thislist.append(key)
    plotted_alignables.append(thislist)
align_outputs=[convertGlobal(align_block,plotted_alignables[0]) for align_block in align_outputs]
title="2022 data, magDown v3 alignment stability tests"
fileprefix="SciFiAlignv3/stability_comparison"

align_empty=calculateDiff(align_outputs[0],align_outputs[0],plotted_alignables[0])
align_survey=makeModulesAlignLogFormat("surveyxml/Modules_surveyInput_20221115.xml",thistype="input")
align_survey_fixed={}
for alignable in align_survey.keys():
    newalignable=alignable.replace("T","FT/T")
    newalignable=newalignable.replace("Q0","HL0/Q0")
    newalignable=newalignable.replace("Q1","HL1/Q1")
    newalignable=newalignable.replace("Q2","HL0/Q2")
    newalignable=newalignable.replace("Q3","HL1/Q3")
    align_survey_fixed[newalignable]=align_survey[alignable]
align_survey=convertGlobal(align_survey_fixed,align_survey_fixed.keys())

plotTxTzMapsGlobal(align_outputs ,files, legendlabels, layers)

# x_data = [-0.1171, -0.1162, -0.268, -0.04142, 0.1621, # Q0
#         0.03502, 0.1165, 0.1497, 0.5838, 0.06308,     # Q2
#         0.2572, 0.3647, -0.1901, -0.18, -0.7523,      # Q1
#         0.5728,0.5549, 0.6089, 0.7914, 0.1837         # Q3
#         ]
# y_data = [2.621e-05, 5.103e-05, 2.416e-04, 1.303e-04, -8.583e-05,
#         -7.568e-05, 3.402e-05, -3.952e-05, -3.085e-05, 4.518e-05,
#         -2.174e-04, -2.855e-04, 1.111e-04, 7.304e-05, 4.857e-04,
#         0.0003424, 0.0004345, 0.0004714, 0.0008559, 0.0006808
#         ]
# x_g = [ -371.71,  -903.7,  -1436.,   -1968.,   -2500.,
#         -160.3,   -692.3,  -1224., -1756.,   -2288.,
#         160.3,    692.3,   1224.,    1756.,    2288.,
#         371.7, 903.7,   1436.,    1968.,    2500.
#         ]
# y_g = [-1208.25, -1208.,   -1208.,   -1208.,   -1208.,    1208.,    1208.,    1208.,
#         1208.,    1208.,   -1208.,   -1208.,   -1208.,   -1208.,   -1208.,    1208.,
#         1208.,    1208.,    1208.,    1208.  ]
#
# plot(x_data, 'tx')
# plot(y_data, 'ty')
# plot(x_g, 'x_global')
