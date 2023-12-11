#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HYPOCENTER-BASED 3D IMAGING OF ACTIVE FAULTS: Utilities for Plotting

Please cite: Truttmann et al. (2023). Hypocenter-based 3D Imaging of Active Faults: Method and Applications in the Southwestern Swiss Alps.

@author: Sandro Truttmann
@contact: sandro.truttmann@gmail.com
@license: GPL-3.0
@date: April 2023
@version: 0.1.1
"""

import numpy as np
import matplotlib
import utilities


def circleplane(p, r, nor):
    """
    Function that calculates the points of a circular plane from the normal
    vector for a given point, radius and normal vector orientation.

    Parameters
    ----------
    p : list
        Coordinates of the EQ epicenter in CH1903+.
    r : int
        Radius of the circle, which is proportional to the magnitude.
    nor : np.array
        Normal vector of the circle plane.

    Returns
    -------
    X, Y, Z coordinates of the circle outline points. These can be used to
    plot a cirucular plane around the EQ epicenter.
    """
    # Calculate two random vectors (v1 & v2) from the normal vector that lie 
    # within the circle plane.
    v1 = np.random.randn(3)                         # take a random vector vx
    v1 -= v1.dot(nor) * nor / np.linalg.norm(nor)**2 # make it orthogonal to nor
    v1 /= np.linalg.norm(v1)                         # normalize it
    v2 = np.cross(nor, v1)                           # cross product with nor

    # Create list of angles in radians
    angles = np.radians(np.arange(0, 361, 30))
        
    # Calculate the outline points using the parametric equation for a circle
    # in 3D
    points = p + r * (np.cos(angles)[:, None] * v1 + np.sin(angles)[:, None] * v2)

    # Append the first point once to ensure that the circle is closed
    points = np.vstack([points, points[0]])

    return points[:, 0], points[:, 1], points[:, 2]


def equal_axes(X, Y, Z):
    """
    This function ensures that 3D (scatter) plots are plotted with REAL equal
    axes.
    As an output, the function returns the ranges of values within the
    DataFrame which can then be used in the plot settings (e.g. axis range).
    """
    # Define the maximal range of values
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
    
    # Define the mid-points for each direction
    mid_x = (max(X)+min(X)) * 0.5
    mid_y = (max(Y)+min(Y)) * 0.5
    mid_z = (max(Z)+min(Z)) * 0.5

    # Output: ranges for X, Y and Z direction to be used in plot settings
    return tuple([[mid - max_range, mid + max_range] for mid in [mid_x, mid_y, mid_z]])


def slipvector_3D(p, r, nor, rake):
    """Calculate 3D vector direction of rake."""

    # Get the xyz coordinates of the horizontal strike vector
    azi, dip = utilities.plane_normal_to_azidip(nor[0], nor[1], nor[2])
    strike = (azi + 90) % 360
    nor_x, nor_y, nor_z = utilities.plane_azidip_to_normal(strike, 90)
    vec_strike = np.array([nor_x, nor_y, 0])
    vec_strike /= np.linalg.norm(vec_strike)     
    
    # Calculate a second in-plane vector (down-dip)
    vec_down = np.cross(nor, vec_strike)
    
    # Precompute sine and cosine of the rake angle
    cos_rake = np.cos(np.radians(-rake))
    sin_rake = np.sin(np.radians(-rake))
    
    # Rotate the vector in-plane according to the rake and get the xyz
    # coordinates
    u = p[0] + r * (cos_rake * vec_strike[0] + sin_rake * vec_down[0])
    v = p[1] + r * (cos_rake * vec_strike[1] + sin_rake * vec_down[1])
    w = p[2] + r * (cos_rake * vec_strike[2] + sin_rake * vec_down[2])
    
    return(u, v, w)


def colorscale(column, cmap, minval, maxval, colorsteps, cmap_reverse=False):
    """
    Define colors for each data entry based on column values.

    Parameters
    ----------
    column : array
        Column after which data should be colored.
    colormap : str
        Name of colormap.
    minval : float / int
        Minimum value of color range.
    maxval : float / int
        Maximum value of color range.
    colorsteps : int
        Number of color steps.

    Returns
    -------
    colors : list
        List of color strings for each data row.

    """
    if cmap_reverse == False:
        cmap = matplotlib.cm.get_cmap(cmap)
    elif cmap_reverse == True:
        cmap = matplotlib.cm.get_cmap(cmap).reversed()
    cmap_list = [cmap(i) for i in np.linspace(0, 1, colorsteps + 1)]
    color_ticks = np.linspace(minval, maxval, len(cmap_list))
    column = np.where(np.isnan(column), -999, column)
    indices = np.digitize(column, color_ticks) - 1
    colors = [f'rgba{cmap_list[idx]}' if idx >= 0 and idx < len(cmap_list) else np.nan for idx in indices]
    colors = [np.nan if c == -999 else c for c in colors]
    
    return colors


def colorscale_mplstereonet(column, cmap, minval, maxval, colorsteps, cmap_reverse=False):
    """
    Define colors for each data entry based on column values for mplstereonet.

    Parameters
    ----------
    column : array
        Column after which data should be colored.
    colormap : str
        Name of colormap.
    minval : float / int
        Minimum value of color range.
    maxval : float / int
        Maximum value of color range.
    colorsteps : int
        Number of color steps.

    Returns
    -------
    colors : list
        List of color strings for each data row.

    """
    if cmap_reverse == False:
        cmap = matplotlib.cm.get_cmap(cmap)
    elif cmap_reverse == True:
        cmap = matplotlib.cm.get_cmap(cmap).reversed()
    cmap_list = [cmap(i) for i in np.linspace(0, 1, colorsteps + 1)]
    color_ticks = np.linspace(minval, maxval, len(cmap_list))
    colors = []
    for i in range(len(column)):
        if column[i] == np.nan:
            colors.append(np.nan)
        else:
            array = np.asarray(color_ticks)
            value = column[i]
            idx = (np.abs(array - value)).argmin()
            color = cmap_list[idx]
            colors.append(color)
    
    return colors


def opacity(column, minval, maxval, steps):
    """
    Define alpha values for each data entry based on column values.
    """
    alpha_list = np.linspace(0.2, 1, steps + 1)
    color_ticks = np.linspace(minval, maxval, len(alpha_list))
    alpha = []
    for i in range(len(column)):
        if column[i] == np.nan:
            alpha.append(np.nan)
        else:
            array = np.asarray(color_ticks)
            value = column[i]
            idx = (np.abs(array - value)).argmin()
            a = alpha_list[idx]
            alpha.append(a)
    return alpha

