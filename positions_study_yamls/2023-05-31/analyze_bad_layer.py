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

files = [\
         "align_logfiles_stability/json_files/parsedlog_256145.json"
]

legendlabels=[\
              "256145"
]

align_outputs=[open_alignment(thisfile) for thisfile in files]
plotted_alignables=[]
for align_block in align_outputs:
    thislist=[]
    for key in align_block.keys():
        if "FT" in key:
            thislist.append(key)
    plotted_alignables.append(thislist)
align_outputs=[convertGlobal(align_block,plotted_alignables[0]) for align_block in align_outputs]

x_glob = get_data(files, 'x_global', align_outputs)
y_glob = get_data(files, 'y_global', align_outputs)
z_glob = get_data(files, 'z_global', align_outputs)

hit_x = get_data(files, 'average_hit_x', align_outputs)
hit_y = get_data(files, 'average_hit_y', align_outputs)
hit_z = get_data(files, 'average_hit_z', align_outputs)

x_Q0M0, y_Q0M0, z_Q0M0 = [], [], []
x_Q0M1, y_Q0M1, z_Q0M1 = [], [], []
x_Q0M2, y_Q0M2, z_Q0M2 = [], [], []
x_Q0M3, y_Q0M3, z_Q0M3 = [], [], []
hit_x_Q0M0, hit_y_Q0M0, hit_z_Q0M0 = [], [], []
hit_x_Q0M1, hit_y_Q0M1, hit_z_Q0M1 = [], [], []
hit_x_Q0M2, hit_y_Q0M2, hit_z_Q0M2 = [], [], []
hit_x_Q0M3, hit_y_Q0M3, hit_z_Q0M3 = [], [], []

for i in range(12):
    # positions
    x_Q0M0.append(x_glob[i][0][0])
    y_Q0M0.append(y_glob[i][0][0])
    z_Q0M0.append(z_glob[i][0][0])

    x_Q0M1.append(x_glob[i][0][1])
    y_Q0M1.append(y_glob[i][0][1])
    z_Q0M1.append(z_glob[i][0][1])

    x_Q0M2.append(x_glob[i][0][2])
    y_Q0M2.append(y_glob[i][0][2])
    z_Q0M2.append(z_glob[i][0][2])

    x_Q0M3.append(x_glob[i][0][3])
    y_Q0M3.append(y_glob[i][0][3])
    z_Q0M3.append(z_glob[i][0][3])

    # hit positions
    hit_x_Q0M0.append(float(hit_x[i][0][0]))
    hit_y_Q0M0.append(float(hit_y[i][0][0]))
    hit_z_Q0M0.append(float(hit_z[i][0][0]))

    hit_x_Q0M1.append(float(hit_x[i][0][1]))
    hit_y_Q0M1.append(float(hit_y[i][0][1]))
    hit_z_Q0M1.append(float(hit_z[i][0][1]))

    hit_x_Q0M2.append(float(hit_x[i][0][2]))
    hit_y_Q0M2.append(float(hit_y[i][0][2]))
    hit_z_Q0M2.append(float(hit_z[i][0][2]))

    hit_x_Q0M3.append(float(hit_x[i][0][3]))
    hit_y_Q0M3.append(float(hit_y[i][0][3]))
    hit_z_Q0M3.append(float(hit_z[i][0][3]))
'''
    i dont understand why x positions are different when they are all
    behind each other (all are Q0)

    y is different in U, V vs X1, X2 since X layers are stereo
'''

correct_order = [2, 0, 1, 3, 6, 4, 5, 7, 10, 8, 9, 11]

correct_x_Q0M0 = [x_Q0M0[i] for i in correct_order]
correct_y_Q0M0 = [y_Q0M0[i] for i in correct_order]
correct_z_Q0M0 = [z_Q0M0[i] for i in correct_order]

correct_x_Q0M1 = [x_Q0M1[i] for i in correct_order]
correct_y_Q0M1 = [y_Q0M1[i] for i in correct_order]
correct_z_Q0M1 = [z_Q0M1[i] for i in correct_order]

correct_x_Q0M2 = [x_Q0M2[i] for i in correct_order]
correct_y_Q0M2 = [y_Q0M2[i] for i in correct_order]
correct_z_Q0M2 = [z_Q0M2[i] for i in correct_order]

correct_x_Q0M3 = [x_Q0M3[i] for i in correct_order]
correct_y_Q0M3 = [y_Q0M3[i] for i in correct_order]
correct_z_Q0M3 = [z_Q0M3[i] for i in correct_order]

correct_hit_x_Q0M0 = [hit_x_Q0M0[i] for i in correct_order]
correct_hit_y_Q0M0 = [hit_y_Q0M0[i] for i in correct_order]
correct_hit_z_Q0M0 = [hit_z_Q0M0[i] for i in correct_order]

correct_hit_x_Q0M1 = [hit_x_Q0M1[i] for i in correct_order]
correct_hit_y_Q0M1 = [hit_y_Q0M1[i] for i in correct_order]
correct_hit_z_Q0M1 = [hit_z_Q0M1[i] for i in correct_order]

correct_hit_x_Q0M2 = [hit_x_Q0M2[i] for i in correct_order]
correct_hit_y_Q0M2 = [hit_y_Q0M2[i] for i in correct_order]
correct_hit_z_Q0M2 = [hit_z_Q0M2[i] for i in correct_order]

correct_hit_x_Q0M3 = [hit_x_Q0M3[i] for i in correct_order]
correct_hit_y_Q0M3 = [hit_y_Q0M3[i] for i in correct_order]
correct_hit_z_Q0M3 = [hit_z_Q0M3[i] for i in correct_order]

fig = plt.figure(figsize=plt.figaspect(0.5))

ax = fig.add_subplot(2, 2, 1, projection='3d')
ax.scatter(correct_x_Q0M0, correct_z_Q0M0, correct_y_Q0M0, color='blue', label='global pos')
ax.scatter(correct_hit_x_Q0M0, correct_hit_z_Q0M0, correct_hit_y_Q0M0, color='red', label='average hit position')
ax.legend()
ax.grid()
ax.set_title('Q0M0: module positions vs hit positions')
ax.set_xlabel('global x')
ax.set_ylabel('global z')
ax.set_zlabel('global y')
print(correct_hit_x_Q0M0)
ax = fig.add_subplot(2, 2, 2, projection='3d')
ax.scatter(correct_x_Q0M1, correct_z_Q0M1, correct_y_Q0M1, color='blue', label='global pos')
ax.scatter(correct_hit_x_Q0M1, correct_hit_z_Q0M1, correct_hit_y_Q0M1, color='red', label='average hit position')
ax.legend()
ax.grid()
ax.set_title('Q0M1: module positions vs hit positions')
ax.set_xlabel('global x')
ax.set_ylabel('global z')
ax.set_zlabel('global y')

ax = fig.add_subplot(2, 2, 3, projection='3d')
ax.scatter(correct_x_Q0M2, correct_z_Q0M2, correct_y_Q0M2, color='blue', label='global pos')
ax.scatter(correct_hit_x_Q0M2, correct_hit_z_Q0M2, correct_hit_y_Q0M2, color='red', label='average hit position')
ax.legend()
ax.grid()
ax.set_title('Q0M2: module positions vs hit positions')
ax.set_xlabel('global x')
ax.set_ylabel('global z')
ax.set_zlabel('global y')

ax = fig.add_subplot(2, 2, 4, projection='3d')
ax.scatter(correct_x_Q0M3, correct_z_Q0M3, correct_y_Q0M3, color='blue', label='global pos')
ax.scatter(correct_hit_x_Q0M3, correct_hit_z_Q0M3, correct_hit_y_Q0M3, color='red', label='average hit position')
ax.legend()
ax.grid()
ax.set_title('Q0M3: module positions vs hit positions')
ax.set_xlabel('global x')
ax.set_ylabel('global z')
ax.set_zlabel('global y')

plt.savefig('SciFiAlignv3/bad_module_Q0M0.pdf')
plt.show()
plt.clf()
