### Snippets for drawing the TxTz alignment positioning plot

import os
import re
import json
import ROOT
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import mplhep as hep
import statistics
import numpy as np
from copy import deepcopy

hep.style.use("LHCb1")
hep.style.use("fira")
hep.style.use("firamath")

regex_typelabel=re.compile("Q")
regex_amodule=re.compile("dPosXYZ")
regex_rmodule=re.compile("dRotXYZ")
labels=["Tx","Ty","Tz","Rx","Ry","Rz"]
positions=["x_global","y_global","z_global"]
trackInfo=["nTracks","nHits"]

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

## Note: if this script fails check that the initial XML definitions and the first object are on different lines. Manually add a newline if needed
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

# files=[\
#        "/interactive_storage/shollitt/STACK/TESTS/HUMBOLDT/SciFiAlignv3/20230429_DD4HEP_rawsurvey_NilsConfig_MTxTzRz/AlignmentResults/parsed_log.json",
#        "/interactive_storage/shollitt/STACK/TESTS/HUMBOLDT/SciFiAlignv3/20230429_DD4HEP_rawSurveyStarter_LGT_MTxTzRz/AlignmentResults/parsed_log.json",
#        "/interactive_storage/shollitt/STACK/TESTS/HUMBOLDT/SciFiAlignv3/20230501_DD4HEP_rawSurveyStarter_LGTMTxTzRzwithTzConstraint/AlignmentResults/parsed_log.json",
#        "/interactive_storage/shollitt/STACK/TESTS/HUMBOLDT/SciFiAlignv3/20230501_DD4HEP_rawSurveyStarter_LGTMTxTzRzwithTxRzTzconstraint/AlignmentResults/parsed_log.json",
#       ]

files = [\
         "align_logfiles_stability/json_files/parsedlog_256145.json", 
         "align_logfiles_stability/json_files/parsedlog_256163.json", 
         "align_logfiles_stability/json_files/parsedlog_256159.json", 
         "align_logfiles_stability/json_files/parsedlog_256030.json",
]

legendlabels=[\
              "run_256145",
              "run_256163",
              "run_256159",
              "run_256030",
]
# old iternums
# iternum=[6,10,12,9]
# i only have the last iteration so only fill zeroes because first iter = iter 0
iternum = [0,0,0,0]
color=["C1","C2","C3","C4",]

align_outputs=[open_alignment(thisfile) for thisfile in files]
plotted_alignables=[]
for align_block in align_outputs:
    thislist=[]
    for key in align_block.keys():
        if "FT" in key:
            thislist.append(key)
    plotted_alignables.append(thislist)
align_outputs=[convertGlobal(align_block,plotted_alignables[0]) for align_block in align_outputs]
title="HalfModule TxTzRz (LooseGoodTracks) alignment"
fileprefix="SciFiAlignv3/CompareConfigs"      


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


for station in ["FT/T1","FT/T2","FT/T3"]:
    for quarterset in [[0,1],[2,3]]:
        plt.figure()
        plotTxTzMapsGlobal(align_empty,stationIn=[station],quarters=quarterset,color="k")
        plotTxTzMapsGlobal(align_survey_fixed,stationIn=[station],quarters=quarterset,color="grey")
        for ii,align_this in enumerate(align_outputs):
            plotTxTzMapsGlobal(align_this,stationIn=[station],quarters=quarterset,color=color[ii],index=iternum[ii])
        plt.title(f"v3 align tests vs design (black), {station}Q{quarterset[0]} (right) and {station}Q{quarterset[1]}(left)")
        plt.savefig(f"SciFiAlignv3/TxTzscatterWithSurvey_{station.split('/')[1]}Q{quarterset[0]}{quarterset[1]}.pdf")