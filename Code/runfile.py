#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HYPOCENTER-BASED 3D IMAGING OF ACTIVE FAULTS
This script provides an example of how to combine the different modules to perform 'hypocenter-based 3D imaging of active faults'.
Each of the modules can be turned on and off by specifing the respective boolean argument (true/false)

Please cite: Truttmann et al. (2023). Hypocenter-based 3D Imaging of Active Faults: Method and Applications in the Southwestern Swiss Alps.

@author: Sandro Truttmann
@contact: sandro.truttmann@geo.unibe.ch
@license: MIT
@date: December 2022
@version: 0.1
"""

# Clear variables
globals().clear()

# Import external modules
import datetime
import time
import numpy as np
import os

# Import modules of the provided toolbox
import fault_network, model_validation, stress_analysis, auto_class, utilities, visualisation


###############################################################################
###############################################################################
###########################    Input parameters     ###########################

###     Hypocenter input file
# hypo_file needs the following columns (similar to hypoDD .reloc-file)
# ID, LAT, LON, DEPTH, X, Y, Z, EX, EY, EZ, YR, MO, DY, HR, MI, SC, MAG, NCCP, NCCS, NCTP, NCTS, RCC, RCT, CID
hypo_file = '/Users/sandro/projects/Hypocenter-based-3D-imaging-of-active-faults/Example_files/StLeonard/hypoDD_StLeonard.txt'        # File location
hypo_sep = '\t'                                                                                                                     # Separator

###     Output directory
out_dir = '/Users/sandro/projects/Hypocenter-based-3D-imaging-of-active-faults'

###     "Fault network reconstruction" module settings
n_mc = 10                     # Nr of Monte Carlo (MC) simulations
r_nn = 100                      # Search radius [m] of nearest neighbor search
dt_nn = 26298                   # Search time window [h]
mag_type = 'ML'                 # Magnitude type: 'ML' or 'Mw'

###     "Model Validation" module settings
# foc_file needs following columns:
# Yr, Mo, Dy, Hr:Mi, Lat, Lon, Z, Mag, A, Strike1, Dip1, Rake1, Strike2, Dip2,
# Rake2, Pazim, Pdip, Tazim, Tdip, Q, Type, Loc
validation_bool = True
foc_file = '/Users/sandro/projects/Hypocenter-based-3D-imaging-of-active-faults/Example_files/StLeonard/FocalMechanisms_StLeonard.csv'
foc_sep = ';'
foc_mag_check = True
foc_loc_check = True

###     "Automatic Classification" module settings
autoclass_bool = True
n_clusters = 2
algorithm = 'vmf_soft'
rotation = True

###     "Fault Stress Analysis" module settings
stress_bool = True
S1_trend = 301
S1_plunge = 23
S3_trend = 43
S3_plunge = 26
stress_R = 0.35
# Define stress magnitudes after Vavrycuk et al. (2014)
S1_mag = 1
S2_mag = 1 - (2*stress_R)
S3_mag = -1
PP = 0
fric_coeff = 0.75


###############################################################################
# Start the timer
start = time.time()
print('')
print('###   HYPOCENTER-BASED 3D IMAGING OF ACTIVE FAULTS   ###')
print('Calculation started...')
print('')

###############################################################################
# Fault network reconstruction
(input_params,
 data_input, data_output,
 per_X, per_Y, per_Z) = fault_network.faultnetwork3D(hypo_file, hypo_sep, out_dir,
                                              n_mc, r_nn, dt_nn,
                                              validation_bool,  foc_file,
                                              foc_sep,
                                              stress_bool, S1_mag, S2_mag,
                                              S3_mag, PP, S1_trend, S1_plunge,
                                              S3_trend, S3_plunge, stress_R,
                                              fric_coeff,
                                              autoclass_bool, mag_type)

###############################################################################
# Model Validation
data_input, data_output = model_validation.focal_validation(input_params,
                                                            data_input,
                                                            data_output,
                                                            foc_mag_check,
                                                            foc_loc_check)

###############################################################################
# Automatic Classification
data_output = auto_class.auto_classification(input_params,
                                             data_output,
                                             n_clusters,
                                             algorithm=algorithm,
                                             rotation=rotation)

###############################################################################
# Fault Stress Analysis
data_output, S2_trend, S2_plunge = stress_analysis.fault_stress(input_params,
                                                                data_output)

###############################################################################
# Visualisation
visualisation.model_3d(input_params, data_input, data_output)
visualisation.faults_stereoplot(input_params, data_output)
visualisation.nmc_histogram(input_params, data_input, per_X, per_Y, per_Z)

###############################################################################
# Save model output data
utilities.save_data(input_params, data_input, data_output, per_X, per_Y, per_Z)

###############################################################################
# Stop the timer
end = time.time()
runtime = datetime.timedelta(seconds=(end - start))
print('')
print('')
print('Calculation done!')
print('Model runtime: ', str(runtime))


