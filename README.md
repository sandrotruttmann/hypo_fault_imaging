# Hypocenter-based 3D Imaging of Active Faults

This repository contains a first version of the method presented in Truttmann et al. (2023) to image active faults in 3D based on relatively relocated hypocenter datasets.

The currently supported version of Python is 3.8

Following dependencies need to be installed by the user:
- FB8 (https://github.com/tianluyuan/sphere)
- scikit-learn (version 0.23.2)
- numba
- obspy
- mplstereonet
- spherecluster (https://github.com/jasonlaska/spherecluster)
    --> replace the files "spherical_kmeans.py" and "von_mises_fisher_mixture.py" with the updated versions thereof, provided in the folder Misc in this repository
