# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 08:55:06 2021

@author: Truttmann
"""


###############################################################################
### Fault network validation functions
###############################################################################

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
    Hr = focal_import['Hr:Mi'].str.split(pat=':', expand=True)[0]
    Mi = focal_import['Hr:Mi'].str.split(pat=':', expand=True)[1]
    df_date = pd.DataFrame({'year': focal_import['Yr'],
                            'month': focal_import['Mo'],
                            'day': focal_import['Dy'],
                            'hour': Hr,
                            'minute': Mi})
    focal_import['Date'] = pd.to_datetime(df_date)
    focal_import = focal_import.sort_values(by='Date')

    # Merge the hypo and focal data
    df = pd.merge_asof(data_input, focal_import, on='Date',
                                tolerance=pd.Timedelta('60s'))

    # Optional: magnitude cross-check
    if foc_mag_check == True:
        # Check if magnitudes of merged data fits
        # Create list to store all indexes with missfitting magnitudes
        mag_missfit = 0.2
        error_idx = []
        for i in range(len(df)):
            if abs(df['MAG'][i] - df['Mag'][i]) > mag_missfit:
                error_idx.append(i)
            else:
                pass

        # Delete the focal data for the respective missfited hypocenter datasets
        for i in error_idx:
            df.iloc[i, len(data_input.iloc[0, :]):] = np.nan
    else:
        pass

    # Optional: Location cross-check
    if foc_loc_check == True:
        # Check if lat & lon & depth are fitting
        for i in range(len(df)):
            if abs(df['LAT'][i] - df['Lat'][i]) > 0.01:
                print('Please check: Missfit in Lat for event with ID ',
                      df.loc[i, 'ID'])
            else:
                pass
        for i in range(len(df)):
            if abs(df['LON'][i] - df['Lon'][i]) > 0.01:
                print('Please check: Missfit in Lon for event with ID',
                      df.loc[i, 'ID'])
            else:
                pass
        for i in range(len(df)):
            if abs(df['DEPTH'][i] - df['Z_y'][i]) > 1:
                print('Please check: Missfit in Depth for event with ID',
                      df.loc[i, 'ID'])
            else:
                pass
    else:
        pass

    return(df)


def focal_validation(input_params, data_input, data_output, foc_mag_check, foc_loc_check):
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
    
    if input_params['validation_bool'][0] == True:
        
        print('')
        print('Fault network validation...')
        
        # Unpack input parameters
        foc_file = input_params['foc_file'][0]
        foc_sep = input_params['foc_sep'][0]
        
        it = len(data_input)    
        
        # Import data and match hypocenter relocations with the focal mechanisms
        data_input = match_hypoDD_focals(data_input, foc_file, foc_sep, foc_mag_check, foc_loc_check)

        # Calculate auxiliary plane from Strike1, Dip1, Rake1
        for k in range(len(data_input)):
            strike1 = data_input.loc[k, 'Strike1']
            dip1 = data_input.loc[k, 'Dip1']
            rake1 = data_input.loc[k, 'Rake1']
            auxiliary_plane = aux_plane(strike1, dip1, rake1)
            
            if np.isnan(auxiliary_plane[0]) == True:
                data_input.loc[k, 'Strike2'] = np.nan
                data_input.loc[k, 'Dip2'] = np.nan
                data_input.loc[k, 'Rake2'] = np.nan

            else:
                data_input.loc[k, 'Strike2'] = int(auxiliary_plane[0])
                data_input.loc[k, 'Dip2'] = int(auxiliary_plane[1])
                data_input.loc[k, 'Rake2'] = int(auxiliary_plane[2])


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
                if angle1 < 90:
                    angle1 = angle1
                elif angle1 > 90:
                    angle1 = 180 - angle1
                if angle2 < 90:
                    angle2 = angle2
                elif angle2 > 90:
                    angle2 = 180 - angle2
    
        ###########################################################################
        # Preferred Focal Plane Selection
                # Choose the mean focal plane with the minimal angular difference
                # to the focal plane (only if mean fault plane is calculated)
                # Select the mean focal plane with the smaller angular difference
                # to the mean fault plane
                angle12 = [angle1, angle2]
                data_output.loc[i, 'epsilon'] = min(angle12)
                if pd.isnull(data_input['Strike1'][i]):
                    pass
                else:
                    data_output.loc[i, 'pref_foc'] = angle12.index(min(angle12))+ 1
    
        ###########################################################################
        
        # Count the number of matched focals
        nr_match = data_input['Strike1'].count()
        print('Number of matched focal mechanisms: ', nr_match)

    else:
        data_output['epsilon'] = np.nan
        

    return(data_input, data_output)