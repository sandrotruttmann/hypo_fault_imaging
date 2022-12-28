# Hypocenter-based 3D Imaging of Active Faults

This repository contains a first version of the method presented in Truttmann et al. (2023) to image active faults in 3D based on relatively relocated hypocenter datasets.

If you use the code in this repository please cite the following references:
Truttmann, S., Diehl, T., Herwegh, M. (2023). Hypocenter-based 3D Imaging of Active Faults: Method and Applications in the Southwestern Swiss Alps. JGR, DOI xxx.
Truttmann, S., Diehl, T., Herwegh, M. (2023). Hypocenter-based 3D Imaging of Active Faults: Method and Applications in the Southwestern Swiss Alps [Dataset]. Zenodo. https://doi.org/10.5281/zenodo.7487904
Truttmann, S. (2023). hypo_fault_imaging [Software]. Github-Zenodo. DOI xxx


The currently supported version of Python is 3.8

Following dependencies need to be installed by the user:
- FB8 (https://github.com/tianluyuan/sphere)
- scikit-learn (version 0.23.2)
- numba
- obspy
- mplstereonet
- spherecluster (https://github.com/jasonlaska/spherecluster)
    --> replace the files "spherical_kmeans.py" and "von_mises_fisher_mixture.py" with the updated versions thereof, provided in the folder Misc in this repository