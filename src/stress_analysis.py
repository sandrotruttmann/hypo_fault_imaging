#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HYPOCENTER-BASED 3D IMAGING OF ACTIVE FAULTS: Fault Stress Analysis Module

Please cite: Truttmann et al. (2023). Hypocenter-based 3D Imaging of Active Faults: Method and Applications in the Southwestern Swiss Alps.

@author: Sandro Truttmann
@contact: sandro.truttmann@geo.unibe.ch
@license: GPL-3.0
@date: December 2022
@version: 1.0
"""

# Import modules
import numpy as np
import mplstereonet
import utilities

def stress_on_plane(S1_mag, S2_mag, S3_mag,
                    S1_trend, S1_plunge,
                    S3_trend, S3_plunge,
                    strike, dip,
                    PP, fric_coeff):
    """
    Calculate the stresses on a plane.

    Parameters
    ----------
    S1_mag : int
        Magnitude of maximum principal stress [MPa].
    S2_mag : int
        Magnitude of intermediate principal stress [MPa].
    S3_mag : int
        Magnitude of minimum principal stress [MPa].
    S1_trend : int
        Trend of maximum principal stress.
    S1_plunge : int
        Plunge of maximum principal stress.
    S2_trend : int
        Trend of intermediate principal stress.
    S2_plunge : int
        Plunge of intermediate principal stress.
    S3_trend : int
        Trend of minimum principal stress.
    S3_plunge : int
        Plunge of minimum principal stress.
    strike : int
        Strike of the plane.
    dip : int
        Dip of the plane.
    PP : int
        Pore fluid pressure [MPa].
    fric_coeff : float
        Friction coefficient.

    Returns
    -------
    Stresses on the plane: effective normal stress, shear stress, rake,
    slip tendency, dilation tendency, trend and plunge of S2.

    """
    # Vectorize S1 and S3
    lon, lat = mplstereonet.line(S1_plunge, S1_trend)
    S1_vec = np.asarray(mplstereonet.stereonet2xyz(lon, lat))[:, 0]
    lon, lat = mplstereonet.line(S3_plunge, S3_trend)
    S3_vec = np.asarray(mplstereonet.stereonet2xyz(lon, lat))[:, 0]

    # Calculate S2 (vectorized and trend/plunge)
    S2_vec = np.cross(S1_vec, S3_vec).round(10)
    S2_plunge, S2_trend = mplstereonet.vector2plunge_bearing(S2_vec[0],
                                                             S2_vec[1],
                                                             S2_vec[2])
    S2_trend = S2_trend[0].round(10)
    S2_plunge = S2_plunge[0].round(10)

    # Define the principal stress coordinate system
    PS = np.array([[S1_mag, 0, 0],
                   [0, S2_mag, 0],
                   [0, 0, S3_mag]
                   ])

    # Convert principal stresses from geographic to xyz orientations
    lon, lat = mplstereonet.line(S1_plunge, S1_trend)
    S1_vec = np.asarray(mplstereonet.stereonet2xyz(lon, lat))[:, 0]
    lon, lat = mplstereonet.line(S2_plunge, S2_trend)
    S2_vec = np.asarray(mplstereonet.stereonet2xyz(lon, lat))[:, 0]
    lon, lat = mplstereonet.line(S3_plunge, S3_trend)
    S3_vec = np.asarray(mplstereonet.stereonet2xyz(lon, lat))[:, 0]

    # Invert z-axis (to make both coordinate systems right-handed)
    S1_vec[2] *= -1
    S2_vec[2] *= -1
    S3_vec[2] *= -1
    
    # Construct the transformation matrix A to convert from principal
    # stress coordinates to geographical coordinate system
    A = np.array([[S1_vec[1], S1_vec[0], S1_vec[2]],
                  [S2_vec[1], S2_vec[0], S2_vec[2]],
                  [S3_vec[1], S3_vec[0], S3_vec[2]]
                  ])
    A = A.round(10)

    # Calculate the stress tensor within the geographic coordinate system and
    # round to 10 decimal numbers
    SG = (A.T @ PS @ A)

    # Define the fault plane coordinate system from fault strike and dip
    strike = np.radians(strike)
    dip = np.radians(dip)
    # Plane normal vector
    nn = np.array([-np.sin(strike) * np.sin(dip),
                   np.cos(strike) * np.sin(dip),
                   -np.cos(dip)
                   ])
    # Plane strike vector
    ns = np.array([np.cos(strike),
                   np.sin(strike),
                   0
                   ])
    # Plane dip vector
    nd = np.array([-np.sin(strike) * np.cos(dip),
                   np.cos(strike) * np.cos(dip),
                   np.sin(dip)
                   ])

    # Project the stress tensor (geographical coordinate system) onto the
    # normal vector of the fault
    t = (SG @ nn)

    # Calculate the normal and shear stresses on the fault plane
    # Total normal stress
    Sn_tot = np.dot(t, nn)
    # Effective normal stress
    Sn_eff = (Sn_tot - PP)
    # Absolute shear stress
    Tau_d = np.dot(t, nd)
    Tau_s = np.dot(t, ns)
    Tau = np.sqrt(Tau_d**2 + Tau_s**2)

    # Calculate the rake (direction of expected fault movement within the fault
    # plane)
    # Negative rake means normal movement, positive rake means reverse movement
    rake = np.arctan2(Tau_d, -Tau_s)
    rake = np.degrees(rake).round(10)
    
    ##############################################################################
    
    # Calculate fault instability (Vavrycuk et al. (2014))
    I = (Tau - fric_coeff*(Sn_eff - 1)) / (fric_coeff + np.sqrt(1 + fric_coeff**2))
    
    ##############################################################################
    
    
    return(Sn_eff, Tau, rake, I, S2_trend, S2_plunge)


def fault_stress(input_params, data_output):
    """
    Calculate the stresses for each individual earthquake.

    Parameters
    ----------
    input_params : DataFrame
        Input parameters.
    data_output : DataFrame
        Output data.

    Returns
    -------
    data_output : DataFrame
        Output data.
    S2_trend : int
        Trend of intermediate principal stress.
    S2_plunge : TYPE
        Plunge of intermediate principal stress.

    """
    
    # Unpack input parameters from dictionary
    for key, value in input_params.items():
        globals()[key] = value

    if input_params['stress_bool'] == True:
        print('')
        print('Fault stress calculation...')
        
        # Define relative stress magnitudes after Vavrycuk et al. (2014)
        S1_mag = 1
        S2_mag = 1 - (2*stress_R)
        S3_mag = -1

            
        # Calculate the stresses on each fault plane
        strike_arr = np.array((data_output['mean_azi'] - 90) % 360)
        dip_arr = np.array(data_output['mean_dip'])
        Sn_eff_list = []
        Tau_list = []
        rake_list = []
        I_list = []
        for i in range(len(data_output)):
            strike = strike_arr[i]
            dip = dip_arr[i]
            Sn_eff, Tau, rake, I, S2_trend, S2_plunge = stress_on_plane(S1_mag,
                                                                      S2_mag,
                                                                      S3_mag,
                                                                      S1_trend,
                                                                      S1_plunge,
                                                                      S3_trend,
                                                                      S3_plunge,
                                                                      strike, dip,
                                                                      PP, fric_coeff)
            Sn_eff_list.append(Sn_eff)
            Tau_list.append(Tau)
            rake_list.append(rake)
            I_list.append(I)
        data_output['Sn_eff'] = Sn_eff_list
        data_output['Tau'] = Tau_list
        data_output['rake'] = rake_list
        data_output['I'] = I_list
        
        
    else:
        S2_trend = np.nan
        S2_plunge = np.nan
        pass

    return data_output, S2_trend, S2_plunge
