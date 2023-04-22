#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HYPOCENTER-BASED 3D IMAGING OF ACTIVE FAULTS: Fault Network Reconstruction Module

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
import scipy as sp
import numba
import multiprocessing as mp
from itertools import zip_longest
from sklearn.neighbors import NearestNeighbors
from sphere.distribution import kent_me
import utilities

@numba.njit
def hypo_perturbation(n_mc, _X, _Y, _Z, EX, EY, EZ, ID):
    """
    Create perturbed hypocenter dataset.

    Parameters
    ----------
    n_mc : int
        Number of perturbations.
    _X : array
        X coordinates.
    _Y : array
        Y coordinates.
    _Z : array
        X coordinates.
    EX : array
        Error in X direction.
    EY : array
        Error in Y direction.
    EZ : array
        Error in Z direction.
    ID : array
        Event IDs.

    Returns
    -------
    Perturbed XYZ hypocenter coordinates in LV95/CH1903+.

    """
    # Create empty arrays for the perturbed hypocenter locations
    per_X = np.empty((_X.shape[0], n_mc))
    per_X[:] = np.nan
    per_Y = np.empty((_Y.shape[0], n_mc))
    per_Y[:] = np.nan
    per_Z = np.empty((_Z.shape[0], n_mc))
    per_Z[:] = np.nan

    # Calculate all perturbed hypocenters
    # Assumption: NORMAL distribution within the error
    # Note: scale parameter (= sigma) is approximately 1/3 of the total
    # error (then it catches 99.7 % of the data according to normal
    # distribution properties)
    for i in range(_X.shape[0]):
        per_X[i,:] = np.random.normal(loc=_X[i], scale=EX[i], size=n_mc)
        per_Y[i,:] = np.random.normal(loc=_Y[i], scale=EY[i], size=n_mc)
        per_Z[i,:] = np.random.normal(loc=_Z[i], scale=EZ[i], size=n_mc)
        
    return per_X, per_Y, per_Z


def nearestneighbors(p, X, r):
    """
    Find the indices and distances of the nearest neighbors to point p in dataset X within a radius r.
    
    Parameters
    ----------
    p : int
        Master event to apply nearest neighbor search on.
    X : array
        Point cloud to be analyzed.
    r : int
        Search radius.

    Returns
    -------
    Indices and distances of points within radius r to point p.

    """
    # Define nearest neighbor parameters
    neigh = NearestNeighbors(radius=r, algorithm='brute', metric='euclidean')

    # Fit the nearest neighbors estimator to the dataset X
    neigh.fit(X)

    # Find the neighbors within a given radius of point p
    rng = neigh.radius_neighbors([p])

    # Save the indices of the points within radius r into NN_idx
    NN_idx = np.asarray(rng[1][0])
    NN_dist = np.asarray(rng[0][0])

    return(NN_idx, NN_dist)


def pca_planefit(X, Y, Z):
    """
    Calculate best fit plane of XYZ point dataset using PCA.

    Parameters
    ----------
    X : array
        X coordinates of points.
    Y : array
        Y coordinates of points.
    Z : array
        Z coordinates of points.

    Returns
    -------
    Principal vectors from PCA (e.g. normal vector) and quality parameters.

    """

    # Best-fit plane
    # Assign the first three rows (XYZ) from the input data to a temporary
    # array for best plane fitting (G)
    G = np.concatenate((X[:, None], Y[:, None], Z[:, None]),
                       axis=1).astype('float64')

    # Calculate the covariance matrix
    cov_matrix = np.cov(G.T)

    # Get the eigenvalues and eigenvectors from the covariance matrix
    eig_vals, eig_vecs = np.linalg.eigh(cov_matrix)

    # Calculate the normal unit vector of the plane, which corresponds to
    # the smallest eigenvector (with the minimal eigenvalue)
    nor = eig_vecs[:, np.argmin(eig_vals)]

    # Calculate the unit vector inside the plane (belonging to lambda 1 & 2)
    v1 = eig_vecs[:, np.argmax(eig_vals)]
    v2 = eig_vecs[:, np.argsort(eig_vals)[1]]
    
    # Convert all normal vector to the lower hemisphere, which means that
    # all z-components of the eigenvectors have to be negative
    # ('point downwards')
    if nor[2] > 0:
        nor = nor * -1

    # Ensure that all normal vectors are normalized to a length of 1
    nor = nor / np.linalg.norm(nor)
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)

    # Plane Fit Robustness Evaluation (based on Jones 2015)
    lam1 = np.max(eig_vals)
    lam2 = np.median(eig_vals)
    lam3 = np.min(eig_vals)
    rat_lam23 = lam2 / lam3
    tot_var = lam1 + lam2 + lam3

    # Define the output of the function
    return (nor, v1, v2, lam1, lam2, lam3, rat_lam23, tot_var)


@numba.njit
def ML_to_MW(ML):
    """
    Convert magnitudes from ML to Mw (after Allmann et al. 2010).

    Parameters
    ----------
    ML : float
        Magnitude ML.

    Returns
    -------
    Magnitude Mw.

    """
    # Use empirical relations from Goertz-Allmann et al. (2011) for Mag transformation
    if ML <= 2:
        Mw = 0.594 * ML + 0.985
    elif ML <= 4:
        Mw = 1.327 + 0.253 * ML + 0.085 * (ML**2)
    elif ML > 4:
        Mw = ML - 0.3

    return Mw


@numba.njit
def faultscalingL14_Mag_A(Mw):
    """
    Convert magnitude to rupture area after Leonard (2014).

    Mw = a + b * log(A)

    Parameters
    ----------
    Mw : int
        Moment magnitude Mw.

    Returns
    -------
    Fault rupture area A (in km2) and the radius r for a cirular fault plane
    (in m).

    """
    # Define the constants a and b for the tectonic setting
    # Constants for SCR SS earthquakes (Leonard 2014, Table 4)
    a = 4.18
    b = 1

    # Calculate the area A from Mw
    A = 10 ** ((Mw - a) / b)

    # Calculate the radius of a circular fault plane
    r = np.sqrt(A / np.pi)     # in km
    r = r * 1000               # in m

    return(A, r)


@numba.njit
def faultscalingL14_Mag_D(Mw):
    """
    Convert magnitude to displacement area after Leonard (2014).

    Mw = a + b * log(D)

    Parameters
    ----------
    Mw : int
        Moment magnitude Mw.

    Returns
    -------
    Fault displacement D (in m).
    """
    # Define the constants a and b for the tectonic setting
    # Constants for SCR SS earthquakes (Leonard 2014, Table 4)
    a = 3.71
    b = 2.0

    # Calculate the magnitude Mw from the fault area
    D = 10 ** ((Mw - a) / b)

    return(D)


@numba.njit
def faultscalingWC94_Mag_A(Mw):
    """
    Convert magnitude to rupture area after Wells & Coppersmith (1994).

    Mw = a + b * log(RA)

    Parameters
    ----------
    Mw : int
        Moment magnitude Mw.

    Returns
    -------
    Fault rupture area A (in km2) and the radius r for a cirular fault plane
    (in m).

    """
    # Define the constants a and b for the tectonic setting
    # Constants for SCR SS earthquakes (Wells & Coppersmith 1994, Table 2A)
    a = 3.98
    b = 1.02

    # Calculate the area A from Mw
    A = 10 ** ((Mw - a) / b)

    # Calculate the radius of a circular fault plane
    r = np.sqrt(A / np.pi)     # in km
    r = r * 1000               # in m

    return(A, r)


@numba.njit
def faultscalingT17_Mag_A(Mw):
    """
    Convert magnitude to rupture area after Thingbaijam (2017).

    Mw = a + b * log(RA)

    Parameters
    ----------
    Mw : int
        Moment magnitude Mw.

    Returns
    -------
    Fault rupture area A (in km2) and the radius r for a cirular fault plane
    (in m).

    """
    # Define the constants a and b for the tectonic setting
    # Constants for SCR SS earthquakes (Thingbaijam 2017, Table 1)
    a = 3.486
    b = 0.942

    # Calculate the area A from Mw
    A = 10 ** ((Mw - a) / b)

    # Calculate the radius of a circular fault plane
    r = np.sqrt(A / np.pi)     # in km
    r = r * 1000               # in m

    return(A, r)


def faultplanes3D(ID, date, X, Y, Z, EX, EY, EZ, r_nn, dt_nn):
    """
    Calculate indiv. fault planes from hypocenter with NN and PCA.

    Parameters
    ----------
    data_input : DataFrame
        Input data (relocated hypocenters and focal mechanisms).
    X : array
        X coordinates in CH1903+ (m).
    Y : array
        Y coordinates in CH1903+ (m).
    Z : array
        Z coordinates in CH1903+ (m).
    r_nn : int
        Search radius for NN search (m).
    dt_nn : int
        Time window for NN search (+- h).
    p_threshold : int
        Minimum number of hypocenters to allow for fault plane calculation.
    lam2_threshold : int
        Collinearity (λ₂) threshold (according to Jones et al. 2015)
    rat_lam23_threshold : int
        Planarity (λ₂ / λ₃) threshold (according to Jones et al. 2015)

    Returns
    -------
    DataFrame with the orientations of the calculated indiv. fault planes.
    """
    # Merge ID and date
    data = np.column_stack((ID, date, X, Y, Z))
    
    ###########################################################################
    # Search nearest neighbors
    NN_idx_list = []
    NN_dist_list = []
    for i in range(len(data)):
        # Execute nearest neighbor search and store the index of the nearest
        # neighbors in a list
        NN_idx, NN_dist = nearestneighbors([data[i, 2], data[i, 3], data[i, 4]
                                            ],
                                           np.array([data[:, 2],
                                                     data[:, 3],
                                                     data[:, 4]]).T,
                                           r_nn)

        # Store indices of nearest neighbors
        NN_idx_list.append(NN_idx)
        NN_dist_list.append(NN_dist)

    # Delete nearest neighbors outside dt_nn threshold
    neigh = []
    for i in range(len(data)):
        # Extract the information from the dataset of the respective rows by
        # the
        NN_idx_i = NN_idx_list[i]
        neigh_i = data[NN_idx_i]
        
        # Get the date of event i (master)
        date_i = np.datetime64(date[i])
            
        # Calculate time delta between event i and its neighbors
        date_j = np.array(neigh_i[:, 1], dtype='datetime64[s]')
        dt_nn_ij = np.abs(date_i - date_j)
        dt_nn_ij = dt_nn_ij.astype('timedelta64[h]')
        dt_nn_ij = dt_nn_ij / np.timedelta64(1, 'h')
        
        # Get the indices of the events j outside the dt_nn threshold
        idx_del = np.where(dt_nn_ij > dt_nn)[0]

        # Delete the neighbouring events j outside dt_nn
        neigh_i = np.delete(neigh_i, idx_del, 0)

        # Append the nearest neighbors of point i to the list 'neigh'
        neigh.append(neigh_i)

    ###########################################################################
    # PCA Plane Fitting

    # Define the plane fitting reliability parameters
    # Minimum number of points to allow for best fit plane calculation
    p_threshold = 5
    # Collinearity (λ₂) (according to Jones et al. 2015)
    # Use the squared mean XYZ relocation error
    lam2_threshold = np.array([EX.mean(), EY.mean(), EZ.mean()]
                              ).mean().round(0)
    lam2_threshold = lam2_threshold**2

    # Planarity (λ₂ / λ₃) (according to Jones et al. 2015)
    rat_lam23_threshold = 5

    # Create an empty array
    plane_fit = np.empty((len(data), 16))
    plane_fit[:] = np.nan

    # Pre-allocate arrays
    X = np.empty(p_threshold)
    Y = np.empty(p_threshold)
    Z = np.empty(p_threshold)
    nor = np.empty(3)
    
    # Loop through all clusters and calculate the best fit plane for each
    # cluster individually
    for i in range(len(data)):
        data_i = neigh[i]

        # Calculate best fit plane if the number of events within the cluster
        # is larger than p_threshold, otherwise insert NaN
        if len(data_i) >= p_threshold:
            X, Y, Z = data_i[:, 2], data_i[:, 3], data_i[:, 4]
            nor, v1, v2, lam1, lam2, lam3, rat_lam23, tot_var = pca_planefit(X, Y, Z)
            # Convert all normal vector to the lower hemisphere, which means
            # that all z-components of the eigenvectors have to be negative
            # ('point downwards')
            nor = nor * np.where(nor[2] > 0, -1, 1)

            # Reliability of fitted planes
            # Check whether the plane meets the defined quality criteria
            # (lam2_threshold and rat_lam23_threshold)
            # If criteria is not fulfilled, insert NaN
            values = np.hstack([data[i, 0], nor, v1, v2, lam1, lam2, lam3, rat_lam23, tot_var, len(data_i)])
            if not (lam2 > lam2_threshold and rat_lam23 > rat_lam23_threshold):
                values[1:15] = np.nan
        else:
            values = np.hstack([data[i, 0], [np.nan] * 14, len(data_i)])
        plane_fit[i] = values

    return(plane_fit)


def faultnetwork3D(input_params):
    """
    Calculate 3D fault network from hypocenters.

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
    DataFrames with the parameters of the input parameters, the full 3D fault
    network model and the MC hypocenter locations.
    """
    
    print('')
    print('Fault network reconstruction...')
    
    # Unpack input parameters from dictionary
    for key, value in input_params.items():
        globals()[key] = value
    
    # Data import
    data_input = pd.read_csv(hypo_file, sep=hypo_sep)
    
    # Extract the date and time information of the hypocenters
    data_input['Date'] = pd.to_datetime(pd.DataFrame({'year': data_input['YR'],
                            'month': data_input['MO'],
                            'day': data_input['DY'],
                            'hour': data_input['HR'],
                            'minute': data_input['MI'],
                            'second': data_input['SC']}))
    data_input['_X'] = data_input['X']
    data_input['_Y'] = data_input['Y']
    data_input['_Z'] = -data_input['Z']
    
    # Create an empty DataFrame to store the output data
    data_output = pd.DataFrame(data_input['ID'])

    # Create n_mc perturbed hypocenter datasets for Monte Carlo simulation
    per_X, per_Y, per_Z = hypo_perturbation(n_mc,
                                            np.array(data_input['_X']),
                                            np.array(data_input['_Y']),
                                            np.array(data_input['_Z']),
                                            np.array(data_input['EX']),
                                            np.array(data_input['EY']),
                                            np.array(data_input['EZ']),
                                            np.array(data_input['ID']))

    # Create dataframes for per_X, per_Y, and per_Z
    df_per_X = pd.concat([data_input['ID'], pd.DataFrame(per_X)], axis=1)
    df_per_Y = pd.concat([data_input['ID'], pd.DataFrame(per_Y)], axis=1)
    df_per_Z = pd.concat([data_input['ID'], pd.DataFrame(per_Z)], axis=1)

    ###########################################################################
    # Fault Plane Orientations for each hypocenter dataset

    # Create empty lists to store all possible plane orientations for each
    # event i
    plane_fit_list = []

    # Calculate the fault plane orientations for all perturbated hypocenter
    # locations k
    ID = np.array(data_input['ID'])
    date = np.array(data_input['Date'].dt.strftime('%Y-%m-%d %H:%H:%S'))
    EX = np.array(data_input['EX'])
    EY = np.array(data_input['EY'])
    EZ = np.array(data_input['EZ'])
    for i in range(n_mc):
        # Extract XYZ locations for each perturbation k
        X = per_X[:, i]
        Y = per_Y[:, i]
        Z = per_Z[:, i]

        # Apply the fault plane calculation function for the respective XYZ
        # dataset and save the results to the list
        plane_fit = faultplanes3D(ID, date, X, Y, Z, EX, EY, EZ, r_nn, dt_nn)
        plane_fit_list.append(plane_fit)

    ###########################################################################
    # Directional Statistic

    def dirstats(plane_fit_list, it, n_mc):
        # Extract the normal unit vectors from all k perturbations for each
        # event i and save them in separate lists
        nor_x_i_list = [[] for _ in range(it)]
        nor_y_i_list = [[] for _ in range(it)]
        nor_z_i_list = [[] for _ in range(it)]
        for k in range(n_mc):
            temp = plane_fit_list[k]
            for i in range(it):
                nor_x_i_list[i].append(temp[i, 1])
                nor_y_i_list[i].append(temp[i, 2])
                nor_z_i_list[i].append(temp[i, 3])
                
        # Calculate the directional statistics parameters of the fault planes
        # from the perturbed hypocenter data
        dirstats_output = np.empty((it, 9))
        dirstats_output[:] = np.nan
        for i in range(it):
            nor_x = np.array(nor_x_i_list[i])
            nor_y = np.array(nor_y_i_list[i])
            nor_z = np.array(nor_z_i_list[i])
            # Nan for all if there are no fits
            if np.isnan(nor_x).all():
                nr_fits = np.nan
                mean_vector = np.array([np.nan, np.nan, np.nan])
                R = np.nan
                N = np.nan
                RN = np.nan
                kappa = np.nan
                beta = np.nan
            # Nan for statistical parameters if there are less than 80% fits
            elif np.count_nonzero(~np.isnan(nor_x)) / n_mc < 0.8:
                nr_fits = np.count_nonzero(~np.isnan(nor_x)) / n_mc
                mean_vector = np.array([np.nan, np.nan, np.nan])
                R = np.nan
                N = np.nan
                RN = np.nan
                kappa = np.nan
                beta = np.nan
            # Directional statistics for all events with more than 80% fits
            else:
                # Calculate direction of the pole to the first plane
                if np.isnan(nor_x).all():
                    v1 = [np.nan, np.nan, np.nan]
                else:
                    v1 = np.array([nor_x[np.isfinite(nor_x)][0], nor_y[np.isfinite(nor_y)][0], nor_z[np.isfinite(nor_z)][0]])
                    v1 /= np.linalg.norm(v1)
                # Check every point in the dataset and swap direction if it
                # lies on the other side of the stereoplot
                # (angular difference larger than 90 degrees)
                for j in range(len(nor_x)):
                    vj = [nor_x[j], nor_y[j], nor_z[j]]
                    vj = vj / np.linalg.norm(vj)
                    if np.linalg.norm(v1 - vj) == 0:
                        angle_deg = np.nan
                    else:
                        angle_deg = np.degrees(np.arccos(np.dot(v1, vj)))
                    if angle_deg > 90:
                        nor_x[j] = nor_x[j] * -1
                        nor_y[j] = nor_y[j] * -1
                        nor_z[j] = nor_z[j] * -1
                
                # Calculate the number of fitted models
                nr_fits = np.count_nonzero(~np.isnan(nor_x)) / n_mc
                # Calculate R and R/N
                v_sum = np.nansum(np.array([nor_x, nor_y, nor_z]), axis=1)
                R = np.linalg.norm(v_sum)
                N = n_mc
                RN = R / N

                # Calculate FB5 (Kent) distribution parameters
                vectors = np.array([nor_x, nor_y, nor_z]).T
                vectors = vectors[~np.isnan(vectors).any(axis=1)]
                try:
                    G = kent_me(vectors)
                    mean_vector = G.gamma1 / np.linalg.norm(G.gamma1)
                    kappa = int(G.kappa)
                    beta = int(G.beta)
                except AssertionError:
                    mean_vector = np.nanmean(vectors, axis=0)
                    kappa = -999
                    beta = -999
                
                # Check if mean lies on upper hemisphere and turn back to lower
                if mean_vector[2] > 0:
                    mean_vector = [-x for x in mean_vector]
                    
                # Loop through plane vectors and make all lower hemisphere again
                for j in range(len(nor_x)):
                    if nor_z[j] > 0:
                        nor_x[j] = -nor_x[j]
                        nor_y[j] = -nor_y[j]
                        nor_z[j] = -nor_z[j]
                        
            # Save the statistical parameters in an array
            dirstats_output[i, :3] = mean_vector
            dirstats_output[i, 3:9] = [nr_fits, R, N, RN, kappa, beta]
    
        return(dirstats_output, nor_x_i_list, nor_y_i_list, nor_z_i_list)


    it = len(data_input)
    dirstats_output, nor_x_i_list, nor_y_i_list, nor_z_i_list = dirstats(plane_fit_list, it, n_mc)

    # Save the directional statistics parameters in the data_output dataframe
    data_output['nor_x_mean'] = dirstats_output[:, 0]
    data_output['nor_y_mean'] = dirstats_output[:, 1]
    data_output['nor_z_mean'] = dirstats_output[:, 2]
    data_output['nr_fits'] = dirstats_output[:, 3]
    data_output['R'] = dirstats_output[:, 4]
    data_output['N'] = dirstats_output[:, 5]
    data_output['R/N'] = dirstats_output[:, 6]
    data_output['kappa'] = dirstats_output[:, 7]
    data_output['beta'] = dirstats_output[:, 8]

    # Set back the negative (=very well-defined) kappa value to the largest value obtained
    max_kappa = data_output['kappa'].max()
    data_output.replace(-999, max_kappa, inplace=True)
    
    # Convert the normal unit vector to azimuth and dip of the mean plane
    nor_columns = ['nor_x_mean', 'nor_y_mean', 'nor_z_mean']
    azi_list, dip_list = [], []
    for nor_x, nor_y, nor_z in zip_longest(*(data_output[col] for col in nor_columns)):        
        # Check if vector of event i is NaN; if yes skip event i, else
        # calculate azimuth and dip of the mean plane
        if all(map(np.isnan, [nor_x, nor_y, nor_z])):
            azi, dip = np.nan, np.nan
        else:
            azi, dip = utilities.plane_normal_to_azidip(nor_x, nor_y, nor_z)
        azi_list.append(azi)
        dip_list.append(dip)
    data_output['mean_azi'] = azi_list
    data_output['mean_dip'] = dip_list

    ###########################################################################
    # Magnitude - Fault Area Scaling

    # Transform ML to Mw (after Allmann et al. 2010)
    if mag_type == 'ML':
        data_input['Mw'] = data_input['MAG'].apply(ML_to_MW)
    elif mag_type == 'Mw':
        data_input['Mw'] = data_input['MAG']
    else:
        print('ERROR: no Magnitude type specified')
    # Calculate the rupture area A (in km2) and the diameter r (in m) of a
    # circular rupture plane (after Leonard 2014)
    A, r = np.vectorize(faultscalingL14_Mag_A)(data_input['Mw'])
    data_output.insert(loc=1, column='A', value=A)
    data_output.insert(loc=2, column='r', value=r)

    ###########################################################################

    return(data_input, data_output, df_per_X, df_per_Y, df_per_Z)
