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

hep.style.use(hep.style.LHCb2)
import matplotlib.patches as mpl_patches

def meta_constructor(loader, node):
   return loader.construct_mapping(node)

yaml.add_constructor('!alignment', meta_constructor)

# folder on pc
path_run_folder = "/mnt/c/Users/Nils/Desktop/Promotion/SciFi/positions_study_yamls/2023-05-31"

# yaml files
path_yaml_file = "/256030/Modules.yml"

num = re.findall(r'\d+', path_yaml_file)
print(num[0])

x = [['a', 'b'], ['1', '2']]
d = [[], []]

a = 0
for i in range(2):
    d[a].append(f'iter {a}')
    a += 1

print(d)
