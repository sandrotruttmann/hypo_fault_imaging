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
import json

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
    
    # Unpack input parameters from dictionary
    for key, value in input_params.items():
        globals()[key] = value

    # Create output folder
    out_path = os.path.join(input_params['out_dir'], 'Model_output')
    os.makedirs(out_path, exist_ok=True)
        
    # Save the input parameters to a .txt-file
    with open(str(out_path + '/input_params.txt'), 'w') as outfile:
        json.dump(input_params, outfile)
        
    # Save the input data to a .txt-file
    data_input.to_csv(out_path + '/data_input.txt', sep='\t')

    # Save the output data to a .txt-file
    data_output.to_csv(out_path + '/data_output.txt', sep='\t')

    # Save the perturbed hypocenters to a .txt-file
    per_X.to_csv(out_path + '/per_X.txt', sep='\t')
    per_Y.to_csv(out_path + '/per_Y.txt', sep='\t')
    per_Z.to_csv(out_path + '/per_Z.txt', sep='\t')

    return