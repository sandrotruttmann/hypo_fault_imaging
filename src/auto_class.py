#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HYPOCENTER-BASED 3D IMAGING OF ACTIVE FAULTS: Auto Classification Module

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
import spherecluster


def auto_classification(input_params, data_output):
    """
    Automatic classification of point cloud based on fault orientations.

    Parameters
    ----------
    input_params : DataFrame
        Input parameters.
    data_output : DataFrame
        Output data.
    n_clusters : int
        Number of expected clusters.
    algorithm : str
        Clustering algorithm.
    rotation : bool
        Data rotation.

    Returns
    -------
    Output DataFrame with class labels.

    """

    # Unpack input parameters from dictionary
    for key, value in input_params.items():
        globals()[key] = value

    if input_params['autoclass_bool']:
        print('')
        print('Automatic classification...')
    
        # !!! IMPORTANT NOTE TO MAKE SPHERECLUSTER WORK!!!
        # Replace the content of the files "spherical_kmeans.py" and
        # "von_mises_fisher_mixture.py" with the code from the following repository:
        # https://github.com/jasonlaska/spherecluster/pull/34/commits/d4b70d50bb57a5f314f5e2a3c7dcb5df21fd4ef8
        
        # Extract XYZ columns and remove NaN values
        data = data_output[['nor_x_mean', 'nor_y_mean', 'nor_z_mean']].dropna()
        X, Y, Z = data['nor_x_mean'], data['nor_y_mean'], data['nor_z_mean']
        data = np.array([X, Y, Z]).T
        
        # Option to cluster subvertical structures
        if rotation:
            # Rotate the data to similar directions if necessary
            # Ensure that all vectors point to similar direction as first entry
            v1 = [data[0][0], data[0][1], data[0][2]]
            v1 = v1 / np.linalg.norm(v1)
            # Check every point in the dataset and swap direction if it
            # lies on the other side of the stereoplot
            # (angular difference larger than 90 degrees)
            for j in range(len(data)):
                vj = [data[j, 0], data[j, 1], data[j, 2]]
                vj = vj / np.linalg.norm(vj)
                if np.linalg.norm(v1 - vj) == 0:
                    angle_deg = np.nan
                else:
                    angle_deg = np.degrees(np.arccos(np.dot(v1, vj)))
                if angle_deg > 90:
                    data[j, 0] = data[j, 0] * -1
                    data[j, 1] = data[j, 1] * -1
                    data[j, 2] = data[j, 2] * -1
                else:
                    pass

        # Apply clustering with the specified estimator
        if algorithm == 'skm':
            # Spherical k-Means clustering
            skm = spherecluster.SphericalKMeans(n_clusters=n_clusters)
            skm.fit(data)
            labels = skm.labels_
            cluster_centers = skm.cluster_centers_                
        elif algorithm == 'vmf_soft':
            # Von Mises Fisher Mixture soft clustering
            vmf_soft = spherecluster.VonMisesFisherMixture(n_clusters=n_clusters, posterior_type='soft')
            vmf_soft.fit(data)
            labels = vmf_soft.labels_
            cluster_centers = vmf_soft.cluster_centers_
        elif algorithm == 'vmf_hard':
            # Von Mises Fisher Mixture hard clustering
            vmf_hard = spherecluster.VonMisesFisherMixture(n_clusters=n_clusters, posterior_type='hard')
            vmf_hard.fit(data)
            labels = vmf_hard.labels_
            cluster_centers = vmf_hard.cluster_centers_
        else:
            # Raise error for unsupported algorithm name
            raise ValueError('Unsupported clustering algorithm: {}'.format(algorithm))
        
        # Add the cluster labels to data_output
        data_output.loc[data_output['nor_x_mean'].notna(), 'class'] = labels
        # Reset the index
        data_output = data_output.reset_index(drop=True)
        
        # Print mean directions of each class
        # Rotate to lower hemisphere if cluster center lies on upper hemisphere
        def rotation_lowerhemi(nor_x, nor_y, nor_z):
            if nor_z > 0:
                nor_x = nor_x * -1
                nor_y = nor_y * -1
                nor_z = nor_z * -1
            else:
                pass
            
            azi, dip = utilities.plane_normal_to_azidip(nor_x, nor_y, nor_z)
            return azi, dip

        for q in range(len(cluster_centers)):
            nor_x = cluster_centers[q][0]
            nor_y = cluster_centers[q][1]
            nor_z = cluster_centers[q][2]
            
            azi, dip = rotation_lowerhemi(nor_x, nor_y, nor_z)
            
            print(f'Mean fault orientation class {q}: ', azi, '/', dip)

            # ## Estimate the confidence angle a95
            # # Extract all planes from class q
            # df_q = data_output.loc[data_output['class'] == q]
            # df_q = df_q.reset_index(drop=True)
            # nor_x_list = np.array(df_q['nor_x_mean'])
            # nor_y_list = np.array(df_q['nor_y_mean'])
            # nor_z_list = np.array(df_q['nor_z_mean'])
            # # Calculate the direction of the pole to the first plane
            # nor_x_1 = nor_x[np.isfinite(nor_x)][0]
            # nor_y_1 = nor_y[np.isfinite(nor_y)][0]
            # nor_z_1 = nor_z[np.isfinite(nor_z)][0]
            # v1 = [nor_x_1, nor_y_1, nor_z_1]
            # v1 = v1 / np.linalg.norm(v1)
            # # Check every point in the dataset and swap direction if it
            # # lies on the other side of the stereoplot
            # # (angular difference larger than 90 degrees)
            # for j in range(len(nor_x_list)):
            #     vj = [nor_x_list[j], nor_y_list[j], nor_z_list[j]]
            #     vj = vj / np.linalg.norm(vj)
            #     if np.linalg.norm(v1 - vj) == 0:
            #         angle_deg = np.nan
            #     else:
            #         angle_deg = np.degrees(np.arccos(np.dot(v1, vj)))
            #     if angle_deg > 90:
            #         nor_x_list[j] = nor_x_list[j] * -1
            #         nor_y_list[j] = nor_y_list[j] * -1
            #         nor_z_list[j] = nor_z_list[j] * -1
            #     else:
            #         pass
            # # Calculate R, N and confidence angle (Borradaile 2003)
            # N = len(df_q)
            # v_sum = np.array([np.nansum(nor_x_list), np.nansum(nor_y_list), np.nansum(nor_z_list)])
            # R = np.linalg.norm(v_sum)
            # p = 0.05            # confidence = 1-p
            # a95 = np.arccos(1-((N-R)/R)*((1/p)**(1/(N-1)-1)))
            
    else:
        data_output['class'] = np.nan

    return(data_output)

