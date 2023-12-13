#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HYPOCENTER-BASED 3D IMAGING OF ACTIVE FAULTS: Model Validation Module

Please cite: Truttmann et al. (2023). Hypocenter-based 3D Imaging of Active Faults: Method and Applications in the Southwestern Swiss Alps.

@author: Sandro Truttmann
@contact: sandro.truttmann@gmail.com
@license: GPL-3.0
@date: April 2023
@version: 0.1.1
"""

# Import modules
import numpy as np
import pandas as pd
import utilities
from obspy.imaging.beachball import aux_plane


def match_hypoDD_focals(data_input, foc_file, foc_sep, foc_mag_check, foc_loc_check):
    """
    Load and match hypoDD and focal data (using event time and magnitude).

    Parameters
    ----------
    hypo_file : str
        Path of hypoDD relocation file.
    hypo_sep : str
        Seperator in hypoDD relocation file.
    foc_file : str
        Path of focal mechanism file.
    foc_sep : str
        Separator in focal mechanism file.

    Returns
    -------
    DataFrame with hypoDD events and matched focal mechanisms.

    """
    
    # Import the focal file
    focal_import = pd.read_csv(foc_file, sep=foc_sep)
    
    # Delete the last columns of the focals
    focal_import = focal_import.iloc[:, 0:22]

    # Extract the date and time information of the events with focals
    # Split the 'Hr:Mi' column in two
    
    # Chek if focal_import['Hr:Mi'] is incorporates seconds
    if len(focal_import['Hr:Mi'][0]) > 8:
        datestring = pd.Series(focal_import['Hr:Mi'].str.split('.', expand=True).astype(str).values.T[0])
        Hr, Mi, Sec = datestring.str.split(':', expand=True).astype(int).values.T
        df_date = pd.DataFrame({'year': focal_import['Yr'],
                                'month': focal_import['Mo'],
                                'day': focal_import['Dy'],
                                'hour': Hr,
                                'minute': Mi,
                                'second': Sec})
        focal_import['Date'] = pd.to_datetime(df_date)
        focal_import = focal_import.sort_values(by='Date')
    elif len(focal_import['Hr:Mi'][0]) == 5:
        Hr, Mi = focal_import['Hr:Mi'].str.split(':', expand=True).astype(int).values.T
        df_date = pd.DataFrame({'year': focal_import['Yr'],
                                'month': focal_import['Mo'],
                                'day': focal_import['Dy'],
                                'hour': Hr,
                                'minute': Mi})
    else:
        ValueError('Please check the time format of the focal file!')
    focal_import['Date'] = pd.to_datetime(df_date)
    focal_import = focal_import.sort_values(by='Date')

    # Merge the hypo and focal data
    df = pd.merge_asof(data_input, focal_import, on='Date',
                                tolerance=pd.Timedelta('60s'))

    # Optional: magnitude cross-check
    if foc_mag_check:
        # Check if magnitudes of merged data fits
        # Calculate magnitude difference
        mag_diff = np.abs(df['MAG'] - df['Mag'].values)
        # Find indexes with missfitting magnitudes
        error_idx = np.where(mag_diff > 0.2)[0]
        # Delete the focal data for the respective missfited hypocenter datasets
        df.iloc[error_idx, len(data_input.iloc[0, :]):] = np.nan

    # Optional: Location cross-check
    if foc_loc_check:
        # Check if lat & lon & depth are fitting
        for i, row in df.iterrows():
            if abs(row['LAT'] - row['Lat']) > 0.01:
                print(f"Please check: Missfit in Lat for event with ID {row['ID']}")
            if abs(row['LON'] - row['Lon']) > 0.01:
                print(f"Please check: Missfit in Lon for event with ID {row['ID']}")
            if abs(row['DEPTH'] - row['Z_y']) > 1:
                print(f"Please check: Missfit in Depth for event with ID {row['ID']}")

    return(df)


def focal_validation(input_params, data_input, data_output):
    """
    Validate the fault network model with the focal mechanism data.

    Parameters
    ----------
    input_params : DataFrame
        Input parameters.
    data_input : DataFrame
        Input data (relocated hypocenters and focal mechanisms).
    data_output : DataFrame
        Output data from the function faultnetwork3D.

    Returns
    -------
    Input data with added focal mechanisms, output data with added angular differences.

    """
    
    # Unpack input parameters from dictionary
    for key, value in input_params.items():
        globals()[key] = value

    if validation_bool:
        
        print('')
        print('Fault network validation...')
        
        # Unpack input parameters
        it = len(data_input)    
        
        # Import data and match hypocenter relocations with the focal mechanisms
        data_input = match_hypoDD_focals(data_input, foc_file, foc_sep, foc_mag_check, foc_loc_check)

        # Calculate auxiliary plane from Strike1, Dip1, Rake1
        for k in range(len(data_input)):
            strike1 = data_input.loc[k, 'Strike1']
            dip1 = data_input.loc[k, 'Dip1']
            rake1 = data_input.loc[k, 'Rake1']
            auxiliary_plane = aux_plane(strike1, dip1, rake1)
            
            if np.isnan(auxiliary_plane[0]):
                data_input.loc[k, ['Strike2', 'Dip2', 'Rake2']] = np.nan
            else:
                data_input.loc[k, ['Strike2', 'Dip2', 'Rake2']] = [int(auxiliary_plane[0]), int(auxiliary_plane[1]), int(auxiliary_plane[2])]


        ###########################################################################
        # Angular Difference Calculation
    
        # Calculate the angular difference between the fault plane orientations
        # from faultnetwork3D and the focal mechanisms
        for i in range(it):
            if pd.isnull(data_output['mean_azi'][i]):
                pass
            else:
                # Calculate the normal unit vector of the fault plane of event i
                nor_fau = [data_output['nor_x_mean'][i],
                           data_output['nor_y_mean'][i],
                           data_output['nor_z_mean'][i]
                           ]
                
                # Calculate the normal unit vectors of the two mean focal plane
                # solutions
                azi1 = data_input['Strike1'][i] + 90 % 360
                dip1 = data_input['Dip1'][i]
                azi2 = data_input['Strike2'][i] + 90 % 360
                dip2 = data_input['Dip2'][i]
                nor_x1, nor_y1, nor_z1 = utilities.plane_azidip_to_normal(azi1, dip1)
                nor_x2, nor_y2, nor_z2 = utilities.plane_azidip_to_normal(azi2, dip2)
                nor_foc1 = [nor_x1, nor_y1, nor_z1]
                nor_foc2 = [nor_x2, nor_y2, nor_z2]
    
                # Calculate the angle between the calculated fault plane and the
                # mean focal planes
                angle1 = utilities.angle_between(nor_fau, nor_foc1)
                angle2 = utilities.angle_between(nor_fau, nor_foc2)
                angle1 = angle1 if angle1 < 90 else 180 - angle1
                angle2 = angle2 if angle2 < 90 else 180 - angle2    
                
        ###########################################################################
        # Preferred Focal Plane Selection
                # Choose the mean focal plane with the minimal angular difference
                # to the focal plane (only if mean fault plane is calculated)
                # Select the mean focal plane with the smaller angular difference
                # to the mean fault plane
                angle12 = [angle1, angle2]
                data_output.loc[i, 'epsilon'] = min(angle12)
                if not pd.isnull(data_input['Strike1'][i]):
                    data_output.loc[i, 'pref_foc'] = angle12.index(min(angle12))+ 1
    
        ###########################################################################
        
        # Count the number of matched focals
        nr_match = data_input['Strike1'].count()
        print('Number of matched focal mechanisms: ', nr_match)

    else:
        data_output['epsilon'] = np.nan
        

    return(data_input, data_output)