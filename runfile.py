"""
xxx Description

"""


# Clear variables
globals().clear()


# Import modules
import datetime
import time
import numpy as np

import fault_network, model_validation, stress_analysis, auto_class, utilities, visualisation

###############################################################################
# TEMPORARY input parameter definition!

# St. Leonard (ALL)
hypo_file = 'C:/Users/Truttmann/Dropbox/PhD/Data/Data_SED/Seq_StLeonard/data/hypoDD/hypoDD_CT+CC_LSQR_Resampling.reloc.csv'
hypo_sep = ';'

# ## St. Leonard (4 outliers deleted)
# hypo_file = 'C:/Users/Truttmann/Dropbox/PhD/Data/Data_SED/Seq_StLeonard/data/hypoDD/hypoDD_CT+CC_LSQR_Resampling.reloc_OUTLIERSDELETED.csv'
# hypo_sep = ';'

# ## Anzere (ALL)
# hypo_file = 'C:/Users/Truttmann/Dropbox/PhD/Data/Data_SED/Seq_Anzere/data/hypoDD/hypoDD_CT+CC_LSQR_Resampling.reloc.consolidatedErrors_modified.csv'
# hypo_sep = ';'

# ### Anzère (5 outliers deleted)
# hypo_file = 'C:/Users/Truttmann/Dropbox/PhD/Data/Data_SED/Seq_Anzere/data/hypoDD/hypoDD_CT+CC_LSQR_Resampling.reloc.consolidatedErrors_modified_OUTLIERSDELETED.csv'
# hypo_sep = ';'

# # Diemtigen sequence
# hypo_file = 'C:/Users/Truttmann/Dropbox/PhD/Data/Data_SED/Simon_2021_Diemtigen/Diemtigen_relocatedEvents.csv'
# hypo_sep = ';'

# # # Synthetic test !!!
# hypo_file = 'C:/Users/Truttmann/Dropbox/PhD/Python/03_ExperimentalCode/3D_SeismicFaults/05_SyntheticFaults/SyntheticFaults_100m_NEW_withuncertainty_10_100_10%noise.csv'
# hypo_sep = '\t'

# # Gisborne NZ
# hypo_file = 'C:/Users/Truttmann/Dropbox/PhD/CSIRO_Project/NZ_earthquakes/Data/Eberhardt_2022/hypnzw23R2001_2011_GISB_A_hypofile.csv'
# hypo_sep = ','


out_dir = 'C:/Users/Truttmann/Dropbox/PhD/Publications/Paper1_3DFaultNetworks/Submission_2/PythonCode_Output/'
# out_dir = 'C:/Users/Truttmann/Dropbox/PhD/CSIRO_Project/NZ_earthquakes/hypo_FI_Output'

n_mc = 100
r_nn = 100
dt_nn = 26298
mag_type = 'ML'


validation_bool = True
# foc_file = 'C:/Users/Truttmann/Dropbox/PhD/Publications/Paper1_3DFaultNetworks/Data/FocalMechanisms_Anzere.csv'
foc_file = 'C:/Users/Truttmann/Dropbox/PhD/Publications/Paper1_3DFaultNetworks/Data/FocalMechanisms_StLeonard.csv'
foc_sep = ';'
foc_mag_check = True
foc_loc_check = True

stress_bool = True
# Define stress field orientation and shape ratio R (Kastrup et al. (2004))
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


autoclass_bool = True
n_clusters = 2
algorithm = 'vmf_soft'
rotation = True

# ###############################################################################
# # Define input parameters
# hypo_file = 'C:/Users/Truttmann/Dropbox/PhD/Data/Data_SED/Seq_Anzere/data/hypoDD/hypoDD_CT+CC_LSQR_Resampling.reloc.consolidatedErrors_modified_OUTLIERSDELETED.csv'
# hypo_sep = ';'
# out_dir = 'C:/Users/Truttmann/Dropbox/PhD/Python/03_ExperimentalCode/3D_SeismicFaults/Output_Model'
# n_mc = 100
# r_nn = 600
# dt_nn = 12
# validation_bool = True
# foc_file = 'C:/Users/Truttmann/Dropbox/PhD/Data/Data_SED/00_CH/Nico_seistec_lst_1976_2021_copy.csv'
# foc_sep = ';'
# stress_bool = True
# S1_mag = np.nan
# S2_mag = np.nan
# S3_mag = np.nan
# PP = np.nan
# S1_trend = 125
# S1_plunge = 18
# S3_trend = 27
# S3_plunge = 22
# stress_R = 0.85
# TS_norm_bool = True
# fric_coeff = 0.75
# autoclass_bool = True


### Important information for documentation!!!
# hypo_file needs following columns (similar to hypoDD .reloc-file)
# ID, LAT, LON, DEPTH, X, Y, Z, EX, EY, EZ, YR, MO, DY, HR, MI, SC, MAG, NCCP, NCCS, NCTP, NCTS, RCC, RCT, CID
# foc_file needs following columns:
# Yr, Mo, Dy, Hr:Mi, Lat, Lon, Z, Mag, A, Strike1, Dip1, Rake1, Strike2, Dip2,
# Rake2, Pazim, Pdip, Tazim, Tdip, Q, Type, Loc


# ##############################################################################
# # Launch GUI
# (hypo_file, hypo_sep, out_dir,
#  n_mc, r_nn, dt_nn, validation_bool,
#  foc_file, foc_sep, stress_bool, S1_mag, S2_mag, S3_mag,
#  PP, S1_trend, S1_plunge, S3_trend, S3_plunge, stress_R,
#  TS_norm_bool, fric_coeff, autoclass_bool) = utilities.launch_GUI()

###############################################################################
# Start the timer
start = time.time()
print('MAF3D calculations running...')

###############################################################################
# Fault network model
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
# Fault network validation
data_input, data_output = model_validation.focal_validation(input_params,
                                                            data_input,
                                                            data_output,
                                                            foc_mag_check,
                                                            foc_loc_check)

###############################################################################
# Fault stresses
data_output, S2_trend, S2_plunge = stress_analysis.fault_stress(input_params,
                                                                data_output)

###############################################################################
# Automatic classification
data_output = auto_class.auto_classification(input_params,
                                             data_output,
                                             n_clusters,
                                             algorithm=algorithm,
                                             rotation=rotation)

###############################################################################
# Visualisation
visualisation.model_3d(input_params, data_input, data_output)
visualisation.faults_stereoplot(input_params, data_output)
# visualisation.nmc_histogram(input_params, data_input, per_X, per_Y, per_Z)

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


