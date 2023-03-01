# Hypocenter-based 3D Imaging of Active Faults

The code in this repository allows to image active faults in 3D based on relatively relocated hypocenter datasets after the method presented by Truttmann et al. (2023).

## Scientific publication
If you use the code in this repository please cite the following scientific publication:
- Truttmann, S., Diehl, T., Herwegh, M. (2023). Hypocenter-based 3D Imaging of Active Faults: Method and Applications in the Southwestern Swiss Alps. JGR, DOI xxx.

## Installation
To make this code work on your machine you can simply clone this repository. 

## Requirements
The following dependencies need to be installed by the user (we recommend using conda):
- numba
- obspy
- scikit-learn (version 0.23.2)
- FB8 (https://github.com/tianluyuan/sphere)
- mplstereonet (https://github.com/joferkington/mplstereonet)
- spherecluster (https://github.com/jasonlaska/spherecluster)
    - IMPORTANT: replace the files "spherical_kmeans.py" and "von_mises_fisher_mixture.py" with the updated versions thereof, provided in the folder Misc in this repository
- mycolorpy (https://github.com/binodbhttr/mycolorpy) (only needed for the sensitivity analysis script "parameter_sensitivity.py")

The currently supported version of Python is 3.8

## Modules
The five main modules are each implemented in a separate python file:
- Fault network reconstruction: fault_network.py
- Model validation: model_validation.py
- Automatic classification: auto_class.py
- Fault stress analysis: stress_analysis.py
- Visualisation: visualisation.py
The scripts "utilities.py" and "utilities_plot.py" provide generic functions for some of the calculations in the main modules.

## Usage
The file "runfile.py" in the folder "Code" provides an example of how to run a fault network analysis as presented in Truttmann et al. (2023), combining the different modules.
Since the choice of the input parameters r_nn and dt_nn is critical, an example file for the sensitivity analysis of these parameters is provided in the file "parameter_sensitivity.py" (for details see Truttmann et al. (2023)).
