#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HYPOCENTER-BASED 3D IMAGING OF ACTIVE FAULTS
This script provides an example of how to combine the different modules to perform 'hypocenter-based 3D imaging of active faults'.
Each of the modules can be turned on and off by specifing the respective boolean argument (true/false)

Please cite: Truttmann et al. (2023). Hypocenter-based 3D Imaging of Active Faults: Method and Applications in the Southwestern Swiss Alps.

@author: Sandro Truttmann
@contact: sandro.truttmann@gmail.com
@license: GPL-3.0
@date: April 2023
@version: 0.1.1
"""

# Clear variables
globals().clear()

# Import external modules
import datetime
import time
import numpy as np
import os
import sys

# Import modules of the provided toolbox
sys.path.insert(0, './src')
import fault_network, model_validation, stress_analysis, auto_class, utilities, visualisation


# ##########################    Input parameters     ###########################
input_params = {
    ###     General settings
    'project_title' : 'Northern Valais',                               # Project title
    ###     Hypocenter input file
    'hypo_file' : '/Users/sandro/Library/CloudStorage/Dropbox/PhD/Data/Data_SED/01_Valais/V0/hyporelocation_V0_Valais_preproc.csv',        # File location
    'hypo_sep' : ';',                                                 # Separator
    ###     Output directory
    'out_dir' : os.getcwd(),
    ###     "Fault network reconstruction" module settings
    'n_mc' : 1,                      # number of Monte Carlo simulations; set n_mc = 1 to turn off Monte Carlo simulations 
    'r_nn' : 600,                       # search radius [m] of nearest neighbor search
    'dt_nn' : 8766,                    # search time window [h]
    'mag_type' : 'ML',                  # magnitude type: 'ML' or 'Mw'
    'DBSCAN_outliers' : True,           # detect outliers with DBSCAN clustering (recommended for regional datasets)
    'max_dist' : 500,                  # DBSCAN: maximum distance [m] between two samples for one to be considered as in the neighborhood of the other
    'min_samples' : 30,                 # DBSCAN: number of samples in a neighborhood for a point to be considered as a core point
    'clust_alg' : 'auto',               # DBSCAN: The algorithm to be used by the NearestNeighbors module to compute pointwise distances and find nearest neighbors
    'leaf_size' : 30,                   # DBSCAN: Leaf size passed to BallTree or cKDTree
    ###     "Model Validation" module settings
    'validation_bool' : True,
    'foc_file' : '/Users/sandro/Library/CloudStorage/Dropbox/PhD/Data/Data_SED/01_Valais/V0/focals_V0_Valais_preproc.csv',
    'foc_sep' : ';',
    'foc_mag_check' : True,             # check focal magnitude (recommended)
    'foc_loc_check' : True,             # check focal location (recommended)
    ###     "Automatic Classification" module settings
    'autoclass_bool' : False,
    'n_clusters' : 2,                   # number of expected classes
    'algorithm' : 'vmf_soft',           # clustering algorithm
    'rotation' : True,                  # rotate poles before analysis (recommended for vertical faults)
    ###     "Fault Stress Analysis" module settings
    'stress_bool' : True,
    'S1_trend' : 130,                   # σ1 trend
    'S1_plunge' : 23,                   # σ1 plunge
    'S3_trend' : 34,                    # σ3 trend
    'S3_plunge' : 12,                   # σ3 plunge
    'stress_R' : 0.30,                  # Stress shape ratio
    'PP' : 0,                           # Pore pressure
    'fric_coeff' : 0.75                 # Friction coefficient
}

###############################################################################
# Start the timer
start = time.time()
print('')
print('###   HYPOCENTER-BASED 3D IMAGING OF ACTIVE FAULTS   ###')
print('Calculation started...')
print('')

###############################################################################
# Fault network reconstruction
(data_input, data_input_outliers, data_output,
 per_X, per_Y, per_Z) = fault_network.faultnetwork3D(input_params)
 
print(len(data_input))
print(len(data_input_outliers))
print(len(data_output))
###############################################################################
# Model Validation
data_input, data_output = model_validation.focal_validation(input_params,
                                                            data_input,
                                                            data_output)

###############################################################################
# Automatic Classification
data_output = auto_class.auto_classification(input_params,
                                             data_output)

###############################################################################
# Fault Stress Analysis
data_output, S2_trend, S2_plunge = stress_analysis.fault_stress(input_params,
                                                                data_output)

###############################################################################
# Visualisation
visualisation.model_3d(input_params, data_input, data_input_outliers, data_output)
visualisation.faults_stereoplot(input_params, data_output)
# visualisation.nmc_histogram(input_params, data_input, per_X, per_Y, per_Z)

###############################################################################
# Save model output data
utilities.save_data(input_params, data_input, data_input_outliers, data_output,
                    per_X, per_Y, per_Z)

###############################################################################
# Stop the timer
end = time.time()
runtime = datetime.timedelta(seconds=(end - start))
print('')
print('')
print('Calculation done!')
print('Model runtime: ', str(runtime))


