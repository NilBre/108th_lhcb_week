###############################################################################
# (c) Copyright 2022 CERN for the benefit of the LHCb Collaboration           #
#                                                                             #
# This software is distributed under the terms of the GNU General Public      #
# Licence version 3 (GPL Version 3), copied verbatim in the file "COPYING".   #
#                                                                             #
# In applying this licence, CERN does not waive the privileges and immunities #
# granted to it by virtue of its status as an Intergovernmental Organization  #
# or submit itself to any jurisdiction.                                       #
###############################################################################
"""
Configuration file to test running on data.
"""

from Moore import options, run_moore
from RecoConf.reconstruction_objects import reconstruction
from RecoConf.hlt2_global_reco import reconstruction as hlt2_reconstruction, make_light_reco_pr_kf_without_UT, make_light_reconstruction
from Hlt2Conf.lines import all_lines
import os
import re
from RecoConf.hlt2_tracking import (
    make_PrKalmanFilter_noUT_tracks, make_PrKalmanFilter_Seed_tracks,
    make_PrKalmanFilter_Velo_tracks, make_TrackBestTrackCreator_tracks,
    get_UpgradeGhostId_tool_no_UT, make_hlt2_tracks)
from PyConf.Algorithms import VeloRetinaClusterTrackingSIMD, VPRetinaFullClusterDecoder, PrKalmanFilter, PrKalmanFilter_Downstream, PrKalmanFilter_noUT
from RecoConf.hlt1_tracking import (
    make_VeloClusterTrackingSIMD,     make_RetinaClusters,
    get_global_measurement_provider, make_velo_full_clusters, make_reco_pvs,
    make_PatPV3DFuture_pvs)

from RecoConf.hlt1_muonid import make_muon_hits
from RecoConf.calorimeter_reconstruction import make_digits, make_calo
from PyConf.application import configure_input
from PRConfig.FilesFromDirac import get_access_urls
from DDDB.CheckDD4Hep import UseDD4Hep
from Configurables import LHCb__Det__LbDD4hep__DD4hepSvc as DD4hepSvc

options.input_type = 'MDF'
options.input_raw_format = 0.5
# options.input_type = 'MDF' # look into eos for correct data type
options.simulation = False # in DD4Hep is False the correct option
options.data_type = 'Upgrade'

# set DDDB and CondDB info
options.geometry_version = "trunk"
#CONDDBTag = "nibreer_joint_analysis"
CONDDBTag = "master"
options.conditions_version = CONDDBTag

online_cond_path = '/group/online/hlt/conditions.run3/lhcb-conditions-database'
if os.path.exists(online_cond_path):
    DD4hepSvc().ConditionsLocation = 'file://' + online_cond_path

#cond_path = '/scratch/nibreer/my_conditions_database/lhcb-conditions-database/'
#if os.path.exists(cond_path):
#    DD4hepSvc().ConditionsLocation = 'file://' + cond_path

from glob import glob
from Gaudi.Configuration import *
from GaudiConf import IOHelper

# run 269045
files = glob("/calib/align/LHCb/Tracker/0000269045/Run_*.mdf")
options.input_files = files[0:4000]

options.event_store = 'EvtStoreSvc'
options.histo_file = "GoodLongTracks_histo.root"
#options.ntuple_file = "GoodLongTracks_tuple.root"
options.use_iosvc = True  # True for data, False for MC
# multithreading not working while creating tuples
options.n_threads = 35 # set to 1 for tuples, 30 for align V3
options.evt_max = -1

options.scheduler_legacy_mode = False

configure_input(options)

from Humboldt.utils import runAlignment
from Humboldt.alignment_tracking import make_scifi_tracks_and_particles_prkf
from PyConf.Algorithms import PrCheckEmptyTracks, PrStoreUTHitEmptyProducer

alignmentTracks, alignmentPVs, odin, monitors = make_scifi_tracks_and_particles_prkf(
)

from TAlignment.Alignables import Alignables
elements = Alignables()
elements.FTHalfModules("TxRz")
#elements.FTHalfModules("TxTzRxRz")
# add survey constraints
from Configurables import SurveyConstraints
from PyConf.Tools import AlignChisqConstraintTool

surveyconstraints = SurveyConstraints()
surveyconstraints.FT(addHalfModuleJoints=True)

#surveyconstraints.Constraints += [
    # allow for an Rz of ~1mm/1m around the z=0 part of the half module
#    "FT/T./(X1|X2|U|V)/HL./Q./M. :  0 0 0 0 0 0 : 0.1 0.1 0.1 0.001 0.001 0.001 : 0 +1213 0"]


# define Lagrange constraints
constraints = []
constraints.append("BackFramesFixed : FT/T3/X2/HL.*/M. : Tx Rz")
#constraints.append('FixMeanModuleT1 : FT/T1/U/HL.*/M. : Tx : total')
#constraints.append('FixMeanModuleT2 : FT/T2/V/HL.*/M. : Tx: total')
#constraints.append("ModulesGlobalConstraint : FT/T./(X1|X2|U|V)/HL./Q./M. : Tx Tz Rx Rz")

from Humboldt.utils import createAlignUpdateTool, createAlignAlgorithm, getXMLWriterList
with createAlignUpdateTool.bind(
        logFile="alignlog_ft_modules_d0_prkalman.txt"
), createAlignAlgorithm.bind(
        xmlWriters=getXMLWriterList(['VP','FT'], prefix='humb-ft-modules-d0/')):
    runAlignment(
        options,
        surveyConstraints=surveyconstraints,
        lagrangeConstraints=constraints,
        alignmentTracks=alignmentTracks,
        alignmentPVs=alignmentPVs,
#        particles=particles,
        odin=odin,
        elementsToAlign=elements,
        monitorList=monitors,
        usePrKalman=True)
