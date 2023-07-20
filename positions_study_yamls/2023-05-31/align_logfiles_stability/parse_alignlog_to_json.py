import os
import re
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("filename_in")
parser.add_argument("filename_out")
parser.add_argument("filename_out_extra")
args=parser.parse_args()
filename_in=args.filename_in
filename_out=args.filename_out
filename_out_extra=args.filename_out_extra

#filename_in="20210112_StationsLayers_TallRall/alignlog.txt"
#filename_out="20210112_StationsLayers_TallRall/parsedlog.json"
#filename_out_extra="20210112_StationsLayers_TallRall/globalpars.txt"

print(f"Reading: {filename_in}")

# regex per iteration
regex_iteration=re.compile("Iteration*")
regex_leadingline=re.compile("Values of constraint equation*")
regex_convergence=re.compile("Convergence*")
regex_alignchi2=re.compile("Normalised alignment change chisquare*")
regex_error2=re.compile("Error2*")
regex_total_local_delta=re.compile("total local delta chi2*")

# regex per alignable
regex_alignable=re.compile("Alignable*")
regex_position_global=re.compile("Global position*")
regex_tracks=re.compile("Number of tracks/hits/outliers*")
regex_alignpars=re.compile("align pars*")
regex_local_delta=re.compile("local delta chi2*")

alignments={}
alignments["converged"]=[]
alignments["align_changechi2"]=[]
alignments["align_totaldeltachi2"]=[]
alignments["align_error2"]=[]
labels=["Tx","Ty","Tz","Rx","Ry","Rz"]
with open(filename_in,"r") as inputfile:
    with open(filename_out_extra, "w") as globalparsfile:
        thisIter=-1
        writeGlobal=0
        thisObject=""
        for line in inputfile:
            # properties per iteration
            if regex_iteration.search(line):
                thisIter+=1
                continue
            if regex_leadingline.search(line):
                writeGlobal=1
                globalparsfile.write(f"Iteration: {thisIter}\n")
                globalparsfile.write(line)
                continue
            if regex_convergence.search(line):
                writeGlobal=0
                if "Not" in line:
                    alignments["converged"].append(0)
                    print(f"Notice: alignment not converged in iteration {thisIter}")
                    globalparsfile.write(f"Notice: alignment not converged in iteration {thisIter}")
                    continue
                else:
                    alignments["converged"].append(1)
                    continue
            if regex_alignchi2.search(line):
                text,value=line.split(":")
                value=value.strip()
                alignments["align_changechi2"].append(float(value))
                continue
            if regex_error2.search(line):
                text,value=line.split(":")
                value=value.strip()
                alignments["align_error2"].append(float(value))
                continue
            if regex_total_local_delta.search(line):
                text,value=line.split(":")
                value=value.split("/")
                alignments["align_totaldeltachi2"].append(float(value[0].strip())/float(value[1].strip()))
                continue

            if writeGlobal:
                globalparsfile.write(line)

            # properties per alignable
            if regex_alignable.search(line):
                # print('line is:', line)
                text,thisObject=line.split(":")
                thisObject=thisObject.strip()
                if thisObject in alignments.keys():
                    continue
                else:
                    alignments[thisObject]={label:[] for label in labels}
                    alignments[thisObject]["x_global"]=[]
                    alignments[thisObject]["y_global"]=[]
                    alignments[thisObject]["z_global"]=[]
                    alignments[thisObject]["average_hit_x"]=[]
                    alignments[thisObject]["average_hit_y"]=[]
                    alignments[thisObject]["average_hit_z"]=[]
                    alignments[thisObject]["nTracks"]=[]
                    alignments[thisObject]["nHits"]=[]
                    alignments[thisObject]["localDeltaChi2"]=[]
                    continue
            if regex_position_global.search(line):
                textlist=re.split("\(|,|\)",line)
                # print(textlist)
                alignments[thisObject]["x_global"].append(textlist[1])
                alignments[thisObject]["y_global"].append(textlist[2])
                alignments[thisObject]["z_global"].append(textlist[3])
                alignments[thisObject]["average_hit_x"].append(textlist[5])
                alignments[thisObject]["average_hit_y"].append(textlist[6])
                alignments[thisObject]["average_hit_z"].append(textlist[7])
                continue
            if regex_tracks.search(line):
                text,trackvars=line.split(":")
                typeslist=trackvars.split()
                alignments[thisObject]["nTracks"].append(typeslist[0])
                alignments[thisObject]["nHits"].append(typeslist[1])
                continue
            if regex_alignpars.search(line):
                text,alignvars=line.split(":")
                varlist=alignvars.split()
                for (label,alignvar) in zip(labels,varlist):
                    alignments[thisObject][label].append(alignvar)
                continue
            if regex_local_delta.search(line):
                text,deltachi2=line.split(":")
                deltachi2=deltachi2.split("/")
                alignments[thisObject]["localDeltaChi2"].append(float(deltachi2[0])/float(deltachi2[1]))
                continue

f=open(filename_out,"w")
f.write( json.dumps(alignments))
f.close()
