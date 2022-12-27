"""
 =============================================================================
# Collection of Functions for Plotting
#
# Author: Sandro Truttmann (sandro.truttmann@geo.unibe.ch)
# Bern, March 2021
# =============================================================================
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
    angles = list(range(0, 360, 30))
    angles = np.radians(angles)

    # Calculate the outline points using the parametric equation for a circle
    # in 3D
    X = []
    Y = []
    Z = []
    for i in range(len(angles)):
        x = p[0] + r * np.cos(angles[i]) * v1[0] + r * np.sin(angles[i]) * v2[0]
        y = p[1] + r * np.cos(angles[i]) * v1[1] + r * np.sin(angles[i]) * v2[1]
        z = p[2] + r * np.cos(angles[i]) * v1[2] + r * np.sin(angles[i]) * v2[2]
        X.append(x)
        Y.append(y)
        Z.append(z)

    # Append the first point a second time to ensure that the circle is closed
    X.append(p[0] + r * np.cos(angles[0]) * v1[0] + r *
             np.sin(angles[0]) * v2[0])
    Y.append(p[1] + r * np.cos(angles[0]) * v1[1] + r *
             np.sin(angles[0]) * v2[1])
    Z.append(p[2] + r * np.cos(angles[0]) * v1[2] + r *
             np.sin(angles[0]) * v2[2])

    return(X, Y, Z)

def equal_axes(X, Y, Z):
    """
    This function ensures that 3D (scatter) plots are plotted with REAL equal
    axes.
    As an output, the function returns the ranges of values within the
    DataFrame which can then be used in the plot settings (e.g. axis range).
    """
    # Define the maximal range of values
    max_range = np.array([max(X)-min(X),
                          max(Y)-min(Y),
                          max(Z)-min(Z)]
                         ).max() / 2.0

    # Define the mid-points for each direction
    mid_x = (max(X)+min(X)) * 0.5
    mid_y = (max(Y)+min(Y)) * 0.5
    mid_z = (max(Z)+min(Z)) * 0.5

    # Define the range values for each direction
    x_range = [mid_x - max_range, mid_x + max_range]
    y_range = [mid_y - max_range, mid_y + max_range]
    z_range = [mid_z - max_range, mid_z + max_range]

    # Output: ranges for X, Y and Z direction to be used in plot settings
    return(x_range, y_range, z_range)


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
    
    # Rotate the vector in-plane according to the rake and get the xyz
    # coordinates
    rake = np.radians(-rake)
    u = p[0] + r * np.cos(rake) * vec_strike[0] + r * np.sin(rake) * vec_down[0]
    v = p[1] + r * np.cos(rake) * vec_strike[1] + r * np.sin(rake) * vec_down[1]
    w = p[2] + r * np.cos(rake) * vec_strike[2] + r * np.sin(rake) * vec_down[2]

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
    cmap_list = []
    for i in np.linspace(0, 1, colorsteps + 1):
        cmap_list.append(cmap(i))
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
            color = 'rgba' + str(color)
            colors.append(color)
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
    cmap_list = []
    for i in np.linspace(0, 1, colorsteps + 1):
        cmap_list.append(cmap(i))
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

