#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HYPOCENTER-BASED 3D IMAGING OF ACTIVE FAULTS: General utilities and helper function

Please cite: Truttmann et al. (2023). Hypocenter-based 3D Imaging of Active Faults: Method and Applications in the Southwestern Swiss Alps.

@author: Sandro Truttmann
@contact: sandro.truttmann@geo.unibe.ch
@license: GPL-3.0
@date: December 2022
@version: 1.0
"""

# Import modules
import numpy as np
import numba
import pandas as pd
import os


def store_inputparams(hypo_file, hypo_sep, out_dir, n_mc, r_nn, dt_nn, validation_bool, 
                      foc_file, foc_sep, stress_bool, S1_mag, S2_mag, S3_mag, PP,
                      S1_trend, S1_plunge, S3_trend, S3_plunge, stress_R,
                      fric_coeff, autoclass_bool, mag_type):
    """

    Parameters
    ----------
    
    hypo_file : str
        Path of hypoDD input file.
    hypo_sep : str
        Separator for hypoDD input file.
    out_dir : str
        Path for output folder.
    n_mc : int
        Number of MC simulations.
    r_nn : int
        Search radius for nearest neighbor search [m].
    dt_nn : int
        Search time for nearest neighbor search [h].
    validation_bool : bool
        If True: perform model validation calculations.
    foc_file : str
        Path for focal mechanism catalog.
    foc_sep : str
        Separator for focal mechanism catalog.
    stress_bool : bool
        If True: perform fault stress analysis.
    S1_mag : int
        Maximum principal stress magnitude [MPa].
    S2_mag : int
        Intermediate principal stress magnitude [MPa].
    S3_mag : int
        Minimum principal stress magnitude [MPa].
    PP : int
        Pore fluid pressure [MPa].
    S1_trend : int
        Trend of maximum principal stress direction.
    S1_plunge : int
        Plunge of maximum principal stress direction.
    S3_trend : int
        Trend of minimum principal stress direction.
    S3_plunge : int
        Plunge of minimum principal stress direction.
    stress_R : float
        Stress shape ratio R.
    fric_coeff : float
        Friction coefficient.
    autoclass_bool : bool
        If True: perform autoclassification.
    mag_type : str    
        Type of magnitude (ML, Mw)
    

    Returns
    -------
    input_params : DataFrame
        Input parameters.

    """
    input_params = pd.DataFrame(np.nan, index=[0], columns=['hypo_file',
                                                      'hypo_sep',
                                                      'out_dir',
                                                      'n_mc',
                                                      'r_nn',
                                                      'dt_nn',
                                                      'validation_bool',
                                                      'foc_file',
                                                      'foc_sep',
                                                      'stress_bool',
                                                      'S1_trend', 'S1_plunge',
                                                      'S3_trend', 'S3_plunge',
                                                      'stress_R', 'reduced_stress_tens_bool',
                                                      'fric_coeff',
                                                      'autoclass_bool',
                                                      'mag_type'
                                                      ])
    
    input_params['hypo_file'] = hypo_file
    input_params['hypo_sep'] = hypo_sep
    input_params['out_dir'] = out_dir
    input_params['n_mc'] = n_mc
    input_params['r_nn'] = r_nn
    input_params['dt_nn'] = dt_nn
    input_params['validation_bool'] = validation_bool
    input_params['foc_file'] = foc_file
    input_params['foc_sep'] = foc_sep
    input_params['stress_bool'] = stress_bool
    input_params['S1_mag'] = S1_mag
    input_params['S2_mag'] = S2_mag
    input_params['S3_mag'] = S3_mag
    input_params['PP'] = PP
    input_params['S1_trend'] = S1_trend
    input_params['S1_plunge'] = S1_plunge
    input_params['S3_trend'] = S3_trend
    input_params['S3_plunge'] = S3_plunge
    input_params['stress_R'] = stress_R
    input_params['fric_coeff'] = fric_coeff
    input_params['autoclass_bool'] = autoclass_bool
    input_params['mag_type'] = mag_type
    
    # # TODO: change to dictionary!
    # def store_inputparams(**kwargs):
    # input_params = {
    #     'hypo_file': kwargs.get('hypo_file'),
    #     'hypo_sep': kwargs.get('hypo_sep'),
    #     'out_dir': kwargs.get('out_dir'),
    #     'n_mc': kwargs.get('n_mc'),
    #     'r_nn': kwargs.get('r_nn'),
    #     'dt_nn': kwargs.get('dt_nn'),
    #     'validation_bool': kwargs.get('validation_bool'),
    #     'foc_file': kwargs.get('foc_file'),
    #     'foc_sep': kwargs.get('foc_sep'),
    #     'stress_bool': kwargs.get('stress_bool'),
    #     'S1_mag': kwargs.get('S1_mag'),
    #     'S2_mag': kwargs.get('S2_mag'),
    #     'S3_mag': kwargs.get('S3_mag'),
    #     'PP': kwargs.get('PP'),
    #     'S1_trend': kwargs.get('S1_trend'),
    #     'S1_plunge': kwargs.get('S1_plunge'),
    #     'S3_trend': kwargs.get('S3_trend'),
    #     'S3_plunge': kwargs.get('S3_plunge'),
    #     'stress_R': kwargs.get('stress_R'),
    #     'reduced_stress_tens_bool': kwargs.get('reduced_stress_tens_bool'),
    #     'fric_coeff': kwargs.get('fric_coeff'),
    #     'autoclass_bool': kwargs.get('autoclass_bool'),
    #     'mag_type': kwargs.get('mag_type')
    
    return input_params


def trendplunge_to_vector(trend, plunge):
    """
    Convert from trend/plunge to x/y/z coordinate system.

    Parameters
    ----------
    trend : DataFrame
        Trend.
    plunge : DataFrame
        Plunge.

    Returns
    -------
    XYZ coordinates.

    """
    # Convert to radians
    trend = np.radians(trend)
    plunge = np.radians(plunge)
    # Calculate the normal unit vector components
    nor_x = np.sin(0.5 * np.pi - plunge) * np.sin(trend)
    nor_y = np.sin(0.5 * np.pi - plunge) * np.cos(trend)
    nor_z = - np.cos(0.5 * np.pi - plunge)
    
    return(nor_x, nor_y, nor_z)


@numba.njit
def rake_to_trendplunge(plane_strike, plane_dip, rake):
    """
    Convert from Strike-Dip-Rake to lineation trend and plunge.

    Parameters
    ----------
    plane_strike : int
        Plane strike (right-hand rule (RHR)).
    plane_dip : int
        Plane dip.
    rake : int
        Rake in RHR format (0-180° measured from RHR strike direction).

    Returns
    -------
    Trend and plunge.

    """

    # Convert degrees to radians
    S = np.deg2rad(plane_strike)
    D = np.deg2rad(plane_dip)
    R = np.deg2rad(rake)
    
    # Calculate beta in dependence of rake
    beta = abs(np.arctan(np.tan(R) * np.cos(D)))
    beta = np.pi - beta if R > (np.pi / 2) else beta
    
    # Calculate lineation trend and plunge
    trend = S + beta
    plunge = np.arcsin(np.sin(D) * np.sin(R))
    
    # Convert to degrees and round
    trend = int(round(np.degrees(trend))) % 360
    plunge = int(round(np.degrees(plunge)))

    return(trend, plunge)


@numba.njit
def plane_azidip_to_normal(azi, dip):
    """
    Convert plane azimuth and dip (spherical) to normal vector (cartesian).

    Parameters
    ----------
    azi : int
        Plane azimuth in degrees.
    dip : int
        Plane dip in degrees.

    Returns
    -------
    Normal unit vector to the plane in cartesian coordinates.

    """
    # Calculate the orientation of the pole to the plane
    pole_azi = np.radians(azi) + np.pi % 360
    pole_dip = 0.5 * np.pi - np.radians(dip)

    # Calculate the normal unit vector components
    nor_x = np.sin(0.5 * np.pi - pole_dip) * np.sin(pole_azi)
    nor_y = np.sin(0.5 * np.pi - pole_dip) * np.cos(pole_azi)
    nor_z = - np.cos(0.5 * np.pi - pole_dip)

    return(nor_x, nor_y, nor_z)


@numba.njit
def plane_normal_to_azidip(nor_x, nor_y, nor_z):
    """
    Convert plane normal vector (cartesian) to azimuth and dip (cartesian).

    Parameters
    ----------
    nor_x : float
        X component of normal unit vector.
    nor_y : float
        Y component of normal unit vector.
    nor_z : float
        Z component of normal unit vector.

    Returns
    -------
    Plane azimuth and dip .

    """
    # Calulate the plane orientation from the normal vector
    pole_azi = np.arctan2(nor_x, nor_y)
    azi = int(round(np.degrees(pole_azi - np.pi))) % 360

    # Calculate the plane dip from the normal vector
    pole_dip = np.pi - np.arccos(nor_z)
    dip = int(round(np.degrees(pole_dip)))

    return(azi, dip)


def angle_between(v1, v2):
    """
    Return the angle in degrees between two vectors.

    Parameters
    ----------
    v1 : list
        Vector with XYZ components of vector 1.
    v2 : list
        Vector with XYZ components of vector 2.

    Returns
    -------
    Angle between the two vectors (in degrees) .

    """
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    angle = np.degrees(angle)

    return angle


def save_data(input_params, data_input, data_output, per_X, per_Y, per_Z):
    """
    Save data in txt-files.

    Parameters
    ----------
    input_params : DataFrame
        Input parameters.
    data_input : DataFrame
        Input data.
    data_output : DataFrame
        Output data.
    per_X : DataFrame
        X coordinates of MC datasets.
    per_Y : DataFrame
        Y coordinates of MC datasetsION.
    per_Z : DataFrame
        Z coordinates of MC datasets.

    Returns
    -------
    None.

    """
    
    # Create output folder
    out_path = os.path.join(input_params['out_dir'][0], 'Model_output')
    os.makedirs(out_path, exist_ok=True)
        
    # Save the input parameters to a .txt-file
    input_params.to_csv(out_path + '/input_params.txt', sep='\t', index=False)

    # Save the input data to a .txt-file
    data_input.to_csv(out_path + '/data_input.txt', sep='\t')

    # Save the output data to a .txt-file
    data_output.to_csv(out_path + '/data_output.txt', sep='\t')

    # Save the perturbed hypocenters to a .txt-file
    per_X.to_csv(out_path + '/per_X.txt', sep='\t')
    per_Y.to_csv(out_path + '/per_Y.txt', sep='\t')
    per_Z.to_csv(out_path + '/per_Z.txt', sep='\t')

    return