#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HYPOCENTER-BASED 3D IMAGING OF ACTIVE FAULTS
Example script of how to conduct the sensitivity analysis for the input parameters r_nn (search radius) and dt_nn (search time window).

Please cite: Truttmann et al. (2023). Hypocenter-based 3D Imaging of Active Faults: Method and Applications in the Southwestern Swiss Alps.

@author: Sandro Truttmann
@contact: sandro.truttmann@geo.unibe.ch
@license: GPL-3.0
@date: December 2022
@version: 1.0
"""

import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import numpy as np
from mycolorpy import colorlist as mcp
import fault_network, model_validation, stress_analysis

# Specify the different r_nn and dt_nn parameters to be assessed
# (example: St. Leonard)
r_nn_list = [50, 100, 200, 400, 600, 800]
dt_nn_list = [24, 48, 168, 8766, 26298, 999999]

###############################################################################
###############################################################################
###########################    Input parameters     ###########################

###     Hypocenter input file
# hypo_file needs the following columns (similar to hypoDD .reloc-file)
# ID, LAT, LON, DEPTH, X, Y, Z, EX, EY, EZ, YR, MO, DY, HR, MI, SC, MAG, NCCP, NCCS, NCTP, NCTS, RCC, RCT, CID
hypo_file = '/Users/sandro/projects/hypo_fault_imaging/Example_files/StLeonard/hypoDD_StLeonard.txt'        # File location
hypo_sep = '\t'                                                                                                                     # Separator

###     Output directory
out_dir = '/Users/sandro/projects/hypo_fault_imaging'

###     "Fault network reconstruction" module settings
n_mc = 1000                     # Nr of Monte Carlo (MC) simulations
mag_type = 'ML'                 # Magnitude type: 'ML' or 'Mw'

###     "Model Validation" module settings
# foc_file needs following columns:
# Yr, Mo, Dy, Hr:Mi, Lat, Lon, Z, Mag, A, Strike1, Dip1, Rake1, Strike2, Dip2,
# Rake2, Pazim, Pdip, Tazim, Tdip, Q, Type, Loc
validation_bool = True
foc_file = '/Users/sandro/projects/hypo_fault_imaging/Example_files/StLeonard/FocalMechanisms_StLeonard.txt'
foc_sep = ';'
foc_mag_check = True
foc_loc_check = True

###     "Automatic Classification" module settings
autoclass_bool = False

# ###     "Fault Stress Analysis" module settings
stress_bool = False
S1_trend = np.nan
S1_plunge = np.nan
S3_trend = np.nan
S3_plunge = np.nan
stress_R = np.nan
# Define stress magnitudes after Vavrycuk et al. (2014)
S1_mag = 1
S2_mag = 1 - (2*stress_R)
S3_mag = -1
PP = np.nan
fric_coeff = np.nan

# Combine the needed modules of the file "runfile.py" in a function
def runfile_function(r_nn, dt_nn):
    
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
    
    return data_input, data_output, input_params

# Loop over the different combinations of r_nn and dt_nn and plot the cumulative distributions thereof
fig, axs = plt.subplots(nrows=1, ncols=1,
                        figsize=(10, 10))
color_count = 0
cmap = mcp.gen_color(cmap='viridis', n=len(r_nn_list)*len(dt_nn_list))
for i in r_nn_list:
    for j in dt_nn_list:
        now = datetime.now()
        now_time = now.strftime("%H:%M:%S")
        print("Start Time: ", now_time)
        r_nn = i
        dt_nn = j
        print('r_nn: ', r_nn)
        print('dt_nn', dt_nn)
        data_input, data_output, input_params = runfile_function(r_nn, dt_nn)
        
        # Plot data
        data_focals = data_output.dropna(subset=['epsilon'])
        x = (data_focals['epsilon']).sort_values(ascending=True)
        y = pd.Series(range(0, len(x)))
        axs.plot(x, y + 1,
                 label=f'{r_nn}/{dt_nn}',
                 color=cmap[color_count])
        color_count = color_count + 1
            
axs.set_xlabel('Angular difference')
axs.set_ylabel('Cumulative frequency')
axs.legend()

plt.savefig(input_params['out_dir'][0] + '/InputParams_Sensitivity.pdf')