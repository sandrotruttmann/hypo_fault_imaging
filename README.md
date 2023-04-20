

[![DOI](https://zenodo.org/badge/582537470.svg)](https://zenodo.org/badge/latestdoi/582537470)



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
The five main modules are stored in the "src" folder:
- Fault network reconstruction: fault_network.py
- Model validation: model_validation.py
- Automatic classification: auto_class.py
- Fault stress analysis: stress_analysis.py
- Visualisation: visualisation.py
The scripts "utilities.py" and "utilities_plot.py" provide generic functions for some of the calculations in the main modules.

## Usage
The file "runfile.py" in the "samples" folder provides an example of how to run a fault network analysis as presented in Truttmann et al. (2023), conducting the full analysis with all available tools.

Please note:
- The input parameters have to be defined in a dictionary with pre-defined keys (see runfile.py for a template).
- The hypocenter input file follows the hypoDD standard (.reloc file) in terms of header namings. The file specified with "hypo_file" thus needs the following columns and exact names: ID, LAT, LON, DEPTH, X, Y, Z, EX, EY, EZ, YR, MO, DY, HR, MI, SC, MAG, NCCP, NCCS, NCTP, NCTS, RCC, RCT, CID
    - Relocation errors should be given as one standard deviation (σ)
- The focal mechanisms input file needs to be structured according to the following header naming convention: Yr, Mo, Dy, Hr:Mi, Lat, Lon, Z, Mag, A, Strike1, Dip1, Rake1, Strike2, Dip2, Rake2, Pazim, Pdip, Tazim, Tdip, Q, Type, Loc

Since the choice of the input parameters r_nn and dt_nn is critical, an example for the sensitivity analysis of these parameters is provided in the file "inputparams_sensitivity.py" (for details see Truttmann et al. (2023)).
