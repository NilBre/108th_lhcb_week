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
hep.style.use(hep.style.LHCb2)
import matplotlib.patches as mpl_patches

def meta_constructor(loader, node):
   return loader.construct_mapping(node)

yaml.add_constructor('!alignment', meta_constructor)

path_run_folder = "/mnt/c/Users/Nils/Desktop/Promotion/SciFi/108th_lhcb_week/positions_study_yamls/2023-05-31"
# path_yaml_file = "AlignmentResults_Rxunczero_TxRz_constraintaligned/Iter4/yaml/Conditions/FT/Alignment/HalfLayer/Modules.yml"
path_yaml_file = "/256145/Modules.yml"
T1_Tx_yml = []
T1_Tx = []
T1_Rz_yml = []
T1_Rz = []

runs = ["T1UHL0Q0M0", "T1UHL0Q0M1", "T1UHL0Q0M2", "T1UHL0Q0M3", "T1UHL0Q0M4", "T1UHL0Q2M0", "T1UHL0Q2M1", "T1UHL0Q2M2", "T1UHL0Q2M3", "T1UHL0Q2M4", "T1UHL1Q1M0", "T1UHL1Q1M1", "T1UHL1Q1M2", "T1UHL1Q1M3", "T1UHL1Q1M4", "T1UHL1Q3M0", "T1UHL1Q3M1", "T1UHL1Q3M2", "T1UHL1Q3M3", "T1UHL1Q3M4"]

#runs = ["T2UHL0Q0M0", "T2UHL0Q0M1", "T2UHL0Q0M2", "T2UHL0Q0M3", "T2UHL0Q0M4", "T2UHL0Q2M0", "T2UHL0Q2M1", "T2UHL0Q2M2", "T2UHL0Q2M3", "T2UHL0Q2M4", "T2UHL1Q1M0",
# "T2UHL1Q1M1", "T2UHL1Q1M2", "T2UHL1Q1M3", "T2UHL1Q1M4", "T2UHL1Q3M0", "T2UHL1Q3M1", "T2UHL1Q3M2", "T2UHL1Q3M3", "T2UHL1Q3M4"]

#runs = ["T3UHL0Q0M0", "T3UHL0Q0M1", "T3UHL0Q0M2", "T3UHL0Q0M3", "T3UHL0Q0M4", "T3UHL0Q2M0", "T3UHL0Q2M1", "T3UHL0Q2M2", "T3UHL0Q2M3", "T3UHL0Q2M4",
#"T3UHL1Q1M0", "T3UHL1Q1M1", "T3UHL1Q1M2", "T3UHL1Q1M3", "T3UHL1Q1M4", "T3UHL1Q3M0", "T3UHL1Q3M1", "T3UHL1Q3M2", "T3UHL1Q3M3", "T3UHL1Q3M4"]

#T1: UVX1X2
#T2: UVX1X2
#T3: UVX1X2

stations = ["T1", "T2", "T3"]
layers = ["U", "V", "X1", "X2"]

runs_V = []
for i in range(0,len(runs)):
   string = runs[i]
   runs_V.append(string.replace("U", "V"))

runs_X1 = []
for i in range(0,len(runs)):
   string = runs[i]
   runs_X1.append(string.replace("U", "X1"))

runs_X2 = []
for i in range(0,len(runs)):
   string = runs[i]
   runs_X2.append(string.replace("U", "X2"))

x = list(range(len(runs)))

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

for j in range(0,len(stations)):
   for k in range(0,len(layers)):
      if j==0 and k==0:
         #string = runs[j]
         runs_T1_U=runs
      elif j==0 and k==1:
         for i in range(0,len(runs)):
            string = runs[i]
            runs_T1_V.append(string.replace("T1U", "T1V"))
      elif j==0 and k==2:
         for i in range(0,len(runs)):
            string = runs[i]
            runs_T1_X1.append(string.replace("T1U", "T1X1"))
      elif j==0 and k==3:
         for i in range(0,len(runs)):
            string = runs[i]
            runs_T1_X2.append(string.replace("T1U", "T1X2"))
      elif j==1 and k==0:
         for i in range(0,len(runs)):
            string = runs[i]
            runs_T2_U.append(string.replace("T1U", "T2U"))
      elif j==1 and k==1:
         for i in range(0,len(runs)):
            string = runs[i]
            runs_T2_V.append(string.replace("T1U", "T2V"))
      elif j==1 and k==2:
          for i in range(0,len(runs)):
             string = runs[i]
             runs_T2_X1.append(string.replace("T1U", "T2X1"))
      elif j==1 and k==3:
         for i in range(0,len(runs)):
            string = runs[i]
            runs_T2_X2.append(string.replace("T1U", "T2X2"))
      elif j==2 and k==0:
         for i in range(0,len(runs)):
            string = runs[i]
            runs_T3_U.append(string.replace("T1U", "T3U"))
      elif j==2 and k==1:
         for i in range(0,len(runs)):
            string = runs[i]
            runs_T3_V.append(string.replace("T1U", "T3V"))
      elif j==2 and k==2:
          for i in range(0,len(runs)):
            string = runs[i]
            runs_T3_X1.append(string.replace("T1U", "T3X1"))
      elif j==2 and k==3:
         for i in range(0,len(runs)):
            string = runs[i]
            runs_T3_X2.append(string.replace("T1U", "T3X2"))


print(runs_T3_V)

T1U_Tx_yml = []
T1U_Tx = []
T1V_Tx_yml = []
T1V_Tx = []
T1X1_Tx_yml = []
T1X1_Tx = []
T1X2_Tx_yml = []
T1X2_Tx = []

T2U_Tx_yml = []
T2U_Tx = []
T2V_Tx_yml = []
T2V_Tx = []
T2X1_Tx_yml = []
T2X1_Tx = []
T2X2_Tx_yml = []
T2X2_Tx = []

T3U_Tx_yml = []
T3U_Tx = []
T3V_Tx_yml = []
T3V_Tx = []
T3X1_Tx_yml = []
T3X1_Tx = []
T3X2_Tx_yml = []
T3X2_Tx = []

T1U_Rz_yml = []
T1U_Rz = []
T1V_Rz_yml = []
T1V_Rz = []
T1X1_Rz_yml = []
T1X1_Rz = []
T1X2_Rz_yml = []
T1X2_Rz = []

T2U_Rz_yml = []
T2U_Rz = []
T2V_Rz_yml = []
T2V_Rz = []
T2X1_Rz_yml = []
T2X1_Rz = []
T2X2_Rz_yml = []
T2X2_Rz = []

T3U_Rz_yml = []
T3U_Rz = []
T3V_Rz_yml = []
T3V_Rz = []
T3X1_Rz_yml = []
T3X1_Rz = []
T3X2_Rz_yml = []
T3X2_Rz = []

T1U_Tz_yml = []
T1U_Tz = []
T1V_Tz_yml = []
T1V_Tz = []
T1X1_Tz_yml = []
T1X1_Tz = []
T1X2_Tz_yml = []
T1X2_Tz = []

T2U_Tz_yml = []
T2U_Tz = []
T2V_Tz_yml = []
T2V_Tz = []
T2X1_Tz_yml = []
T2X1_Tz = []
T2X2_Tz_yml = []
T2X2_Tz = []

T3U_Tz_yml = []
T3U_Tz = []
T3V_Tz_yml = []
T3V_Tz = []
T3X1_Tz_yml = []
T3X1_Tz = []
T3X2_Tz_yml = []
T3X2_Tz = []

separator = '*'
for i in range(0,len(runs)):
   with open(path_run_folder + path_yaml_file, 'r') as stream:
      data_loaded = yaml.load(stream, Loader=yaml.Loader)
      T1U_Tx_yml.append(data_loaded[runs_T1_U[i]]['position'][0])
      T1U_Tx.append(float(T1U_Tx_yml[i].split(separator, 1)[0]))

      T1V_Tx_yml.append(data_loaded[runs_T1_V[i]]['position'][0])
      T1V_Tx.append(float(T1V_Tx_yml[i].split(separator, 1)[0]))

      T1X1_Tx_yml.append(data_loaded[runs_T1_X1[i]]['position'][0])
      T1X1_Tx.append(float(T1X1_Tx_yml[i].split(separator, 1)[0]))

      T1X2_Tx_yml.append(data_loaded[runs_T1_X2[i]]['position'][0])
      T1X2_Tx.append(float(T1X2_Tx_yml[i].split(separator, 1)[0]))

      T2U_Tx_yml.append(data_loaded[runs_T2_U[i]]['position'][0])
      T2U_Tx.append(float(T2U_Tx_yml[i].split(separator, 1)[0]))

      T2V_Tx_yml.append(data_loaded[runs_T2_V[i]]['position'][0])
      T2V_Tx.append(float(T2V_Tx_yml[i].split(separator, 1)[0]))

      T2X1_Tx_yml.append(data_loaded[runs_T2_X1[i]]['position'][0])
      T2X1_Tx.append(float(T2X1_Tx_yml[i].split(separator, 1)[0]))

      T2X2_Tx_yml.append(data_loaded[runs_T2_X2[i]]['position'][0])
      T2X2_Tx.append(float(T2X2_Tx_yml[i].split(separator, 1)[0]))

      T3U_Tx_yml.append(data_loaded[runs_T3_U[i]]['position'][0])
      T3U_Tx.append(float(T3U_Tx_yml[i].split(separator, 1)[0]))

      T3V_Tx_yml.append(data_loaded[runs_T3_V[i]]['position'][0])
      T3V_Tx.append(float(T3V_Tx_yml[i].split(separator, 1)[0]))

      T3X1_Tx_yml.append(data_loaded[runs_T3_X1[i]]['position'][0])
      T3X1_Tx.append(float(T3X1_Tx_yml[i].split(separator, 1)[0]))

      T3X2_Tx_yml.append(data_loaded[runs_T3_X2[i]]['position'][0])
      T3X2_Tx.append(float(T3X2_Tx_yml[i].split(separator, 1)[0]))

      ###Tz
      T1U_Tz_yml.append(data_loaded[runs_T1_U[i]]['position'][2])
      T1U_Tz.append(float(T1U_Tz_yml[i].split(separator, 1)[0]))

      T1V_Tz_yml.append(data_loaded[runs_T1_V[i]]['position'][2])
      T1V_Tz.append(float(T1V_Tz_yml[i].split(separator, 1)[0]))

      T1X1_Tz_yml.append(data_loaded[runs_T1_X1[i]]['position'][2])
      T1X1_Tz.append(float(T1X1_Tz_yml[i].split(separator, 1)[0]))

      T1X2_Tz_yml.append(data_loaded[runs_T1_X2[i]]['position'][2])
      T1X2_Tz.append(float(T1X2_Tz_yml[i].split(separator, 1)[0]))

      T2U_Tz_yml.append(data_loaded[runs_T2_U[i]]['position'][2])
      T2U_Tz.append(float(T2U_Tz_yml[i].split(separator, 1)[0]))

      T2V_Tz_yml.append(data_loaded[runs_T2_V[i]]['position'][2])
      T2V_Tz.append(float(T2V_Tz_yml[i].split(separator, 1)[0]))

      T2X1_Tz_yml.append(data_loaded[runs_T2_X1[i]]['position'][2])
      T2X1_Tz.append(float(T2X1_Tz_yml[i].split(separator, 1)[0]))

      T2X2_Tz_yml.append(data_loaded[runs_T2_X2[i]]['position'][2])
      T2X2_Tz.append(float(T2X2_Tz_yml[i].split(separator, 1)[0]))

      T3U_Tz_yml.append(data_loaded[runs_T3_U[i]]['position'][2])
      T3U_Tz.append(float(T3U_Tz_yml[i].split(separator, 1)[0]))

      T3V_Tz_yml.append(data_loaded[runs_T3_V[i]]['position'][2])
      T3V_Tz.append(float(T3V_Tz_yml[i].split(separator, 1)[0]))

      T3X1_Tz_yml.append(data_loaded[runs_T3_X1[i]]['position'][2])
      T3X1_Tz.append(float(T3X1_Tz_yml[i].split(separator, 1)[0]))

      T3X2_Tz_yml.append(data_loaded[runs_T3_X2[i]]['position'][2])
      T3X2_Tz.append(float(T3X2_Tz_yml[i].split(separator, 1)[0]))

      ##Rx
      T1U_Rz_yml.append(data_loaded[runs_T1_U[i]]['rotation'][2])
      T1U_Rz.append(float(T1U_Rz_yml[i].split(separator, 1)[0])*1000)

      T1V_Rz_yml.append(data_loaded[runs_T1_V[i]]['rotation'][2])
      T1V_Rz.append(float(T1V_Rz_yml[i].split(separator, 1)[0])*1000)

      T1X1_Rz_yml.append(data_loaded[runs_T1_X1[i]]['rotation'][2])
      T1X1_Rz.append(float(T1X1_Rz_yml[i].split(separator, 1)[0])*1000)

      T1X2_Rz_yml.append(data_loaded[runs_T1_X2[i]]['rotation'][2])
      T1X2_Rz.append(float(T1X2_Rz_yml[i].split(separator, 1)[0])*1000)

      T2U_Rz_yml.append(data_loaded[runs_T2_U[i]]['rotation'][2])
      T2U_Rz.append(float(T2U_Rz_yml[i].split(separator, 1)[0])*1000)

      T2V_Rz_yml.append(data_loaded[runs_T2_V[i]]['rotation'][2])
      T2V_Rz.append(float(T2V_Rz_yml[i].split(separator, 1)[0])*1000)

      T2X1_Rz_yml.append(data_loaded[runs_T2_X1[i]]['rotation'][2])
      T2X1_Rz.append(float(T2X1_Rz_yml[i].split(separator, 1)[0])*1000)

      T2X2_Rz_yml.append(data_loaded[runs_T2_X2[i]]['rotation'][2])
      T2X2_Rz.append(float(T2X2_Rz_yml[i].split(separator, 1)[0])*1000)

      T3U_Rz_yml.append(data_loaded[runs_T3_U[i]]['rotation'][2])
      T3U_Rz.append(float(T3U_Rz_yml[i].split(separator, 1)[0])*1000)

      T3V_Rz_yml.append(data_loaded[runs_T3_V[i]]['rotation'][2])
      T3V_Rz.append(float(T3V_Rz_yml[i].split(separator, 1)[0])*1000)

      T3X1_Rz_yml.append(data_loaded[runs_T3_X1[i]]['rotation'][2])
      T3X1_Rz.append(float(T3X1_Rz_yml[i].split(separator, 1)[0])*1000)

      T3X2_Rz_yml.append(data_loaded[runs_T3_X2[i]]['rotation'][2])
      T3X2_Rz.append(float(T3X2_Rz_yml[i].split(separator, 1)[0])*1000)

# print(T2U_Tx)

fig, ax = plt.subplots(2,2) #, figsize=(10,10))
#fig.canvas.set_window_title('LHCb Internal: Run 264400')
fig.suptitle("LHCb Internal: Run 265583", fontsize=15, weight="bold")

r1 = 215/255
g1 = 48/255
b1 = 39/255
r2 = 252/255
g2 = 141/255
b2 = 89/255

ticksT3 = [runs_T3_U, runs_T3_V,runs_T3_X1, runs_T3_X2]
ypointsT3Tx = [T3U_Tx, T3V_Tx, T3X1_Tx, T3X2_Tx]
ypointsT3Tz = [T3U_Tz, T3V_Tz, T3X1_Tz, T3X2_Tz]
ypointsT3Rz = [T3U_Rz, T3V_Rz, T3X1_Rz, T3X2_Rz]
# print(ypointsT3Tx)

ticksT2 = [runs_T2_U, runs_T2_V,runs_T2_X1, runs_T2_X2]
ypointsT2Tx = [T2U_Tx, T2V_Tx, T2X1_Tx, T2X2_Tx]
ypointsT2Tz = [T2U_Tz, T2V_Tz, T2X1_Tz, T2X2_Tz]
ypointsT2Rz = [T2U_Rz, T2V_Rz, T2X1_Rz, T2X2_Rz]

ticksT1 = [runs_T1_U, runs_T1_V,runs_T1_X1, runs_T1_X2]
ypointsT1Tz = [T1U_Tz, T1V_Tz, T1X1_Tz, T1X2_Tz]
ypointsT1Tx = [T1U_Tx, T1V_Tx, T1X1_Tx, T1X2_Tx]
ypointsT1Rz = [T1U_Rz, T1V_Rz, T1X1_Rz, T1X2_Rz]
#
#T1 Tx Rx
#T2 Tx Rx
#T3 Tx Rx

sum = -1
for i in range(0,2):
   for j in range(0,2):
      sum = sum + 1
      #print(sum)
      plt.sca(ax[i,j])
      plt.xticks(x, ticksT3[sum]) #here
      ax[i,j].scatter(x, ypointsT3Rz[sum], color='red', s=20) ##here
      ax[i,j].tick_params(axis='x', labelrotation = 90, labelsize=12)
      ax[i,j].set_xlabel(r'Module number')
      ax[i,j].set_ylabel(r'T3 modules Rz[mrad]') ##here
      ax[i,j].text(0.1, 0.7, layers[sum], transform=ax[i,j].transAxes, weight="bold")

plt.savefig("T3Rx_halfmod_TxTz_constrainaligned_265583.pdf",bbox_inches='tight') #here
plt.close()
