#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Sandro Truttmann
@contact:
@license
@date:
@version:
"""

###############################################################################
### Auto classification functions
###############################################################################

# Import modules
import numpy as np
import pandas as pd
import utilities
import spherecluster


def auto_classification(input_params, data_output, n_clusters,
                        algorithm='skm', rotation=True):
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

    if input_params['autoclass_bool'][0] == True:
        print('')
        print('Automatic classification...')
    
        # !!! IMPORTANT NOTE TO MAKE SPHERECLUSTER WORK!!!
        # Replace the content of the files "spherical_kmeans.py" and
        # "von_mises_fisher_mixture.py" with the code from the following repository:
        # https://github.com/jasonlaska/spherecluster/pull/34/commits/d4b70d50bb57a5f314f5e2a3c7dcb5df21fd4ef8
        
        # Extract XYZ columns and remove NaN values
        X = data_output['nor_x_mean'].to_numpy()
        Y = data_output['nor_y_mean'].to_numpy()
        Z = data_output['nor_z_mean'].to_numpy()
        X = X[~np.isnan(X)]
        Y = Y[~np.isnan(Y)]
        Z = Z[~np.isnan(Z)]
        data = np.array([X, Y, Z]).T
        
        # Option to cluster subvertical structures
        if rotation == True:
            # Rotate the data to similar directions if necessary
            # Ensure that all vectors point to similar direction as first entry
            v1 = [X[0], Y[0], Z[0]]
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
        
        # Append the class labels to the output DataFrame
        df_labels = pd.DataFrame(columns=[])
        df_labels['index_temp'] = data_output.dropna(subset=['nor_x_mean']).index
        df_labels['class'] = labels
        
        # Add the cluster labels to data_output
        data_output['index_temp'] = data_output.index
        data_output = pd.merge(data_output, df_labels, on='index_temp', how='outer')
        data_output = data_output.drop(columns=['index_temp'])
        
        # Print mean directions of each class
        for q in range(len(cluster_centers)):
            nor_x = cluster_centers[q][0]
            nor_y = cluster_centers[q][1]
            nor_z = cluster_centers[q][2]
            
            # Rotate to lower hemisphere if cluster center lies on upper hemisphere
            if nor_z > 0:
                nor_x = nor_x * -1
                nor_y = nor_y * -1
                nor_z = nor_z * -1
            else:
                pass
            
            azi, dip = utilities.plane_normal_to_azidip(nor_x, nor_y, nor_z)
            
            print(f'Mean fault orientation class {q}: ', azi, '/', dip)

    else:
        data_output['class'] = np.nan

    return(data_output)

