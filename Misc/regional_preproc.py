### Pre-processing of regional relocation data

import pandas as pd
from pyproj import Transformer

in_file = '/Users/sandro/Library/CloudStorage/Dropbox/PhD/Data/Data_SED/01_Valais/V0/01_OriginalData/hyporelocation_Valais_V0_orig.xlsx'
out_path = '/Users/sandro/Library/CloudStorage/Dropbox/PhD/Data/Data_SED/01_Valais/V0/'
# Read in the .xlsx file
df_orig = pd.read_excel(in_file)
print('Number of events in original dataframe: ', len(df_orig))

#######################
# Filtering df_orig
# Only select data if PreMet == 'SVD' or 'LSQ'
df_orig = df_orig[(df_orig['PreMet'] == 'LSQ') | (df_orig['PreMet'] == 'SVD')]

# Only select data with CAT-ID 'DDR-SW0' (Valais)
df_orig = df_orig[df_orig['CAT-ID'] == 'DDR-SW0']

# Only keep tectonic events (labelled with 'T')
df_orig = df_orig[df_orig['T'] == 'T']

# Remove data prior to certain year
df_orig = df_orig[df_orig['YYYY'] >= 1900]

# TODO: only use data with small enough errors? REALLY NEEDED? MAYBE NOT....

# Cross-check if pre-filtering was successful
print('Cross-checking filtered data:')
print(df_orig['PreMet'].unique())
print(df_orig['CAT-ID'].unique())

print('Number of events in original dataframe after filtering: ', len(df_orig))

# Create empty dataframe with pre-defined columns with the necessery input structure
df = pd.DataFrame(columns=['ID', 'LAT', 'LON', 'DEPTH', 'X', 'Y', 'Z', 'EX', 'EY', 'EZ',
                           'YR', 'MO', 'DY', 'HR', 'MI', 'SC', 'MAG', 'NCCP', 'NCCS', 'NCTP',
                           'NCTS', 'RCC', 'RCT', 'CID'])

# Tranform coordinates from lat/lon (WGS84) to CH1903+ (EPSG:21781) using pyproj
# x, y = pyproj.transform(pyproj.Proj(init='epsg:4326'), pyproj.Proj(init='epsg:21781'), df_orig['Pref-lon'].values, df_orig['Pref-lat'].values)
transformer = Transformer.from_crs("epsg:4326", "epsg:21781")
x, y = transformer.transform(df_orig['Pref-lat'].values, df_orig['Pref-lon'].values)

# TODO: check with Tobias if right columns are used here!!!
# Fill the new dataframe with the data from the original dataframe
df['ID'] = df_orig['#DD-ID']
df['LAT'] = df_orig['Pref-lat']
df['LON'] = df_orig['Pref-lon']
df['DEPTH'] = df_orig['Pref-dep']
df['X'] = x     # @Tobias: which colum should we use?
df['Y'] = y     # @Tobias: which colum should we use?
df['Z'] = df_orig['Pref-Z']     # @Tobias: which colum should we use?
df['EX'] = df_orig['Err-X-m']
df['EY'] = df_orig['Err-Y-m']
df['EZ'] = df_orig['Err-Z-m']
df['YR'] = df_orig['YYYY']
df['MO'] = df_orig['MM']
df['DY'] = df_orig['DD']
df['HR'] = df_orig['HH']
df['MI'] = df_orig['MI']
df['SC'] = df_orig['SS']
df['MAG'] = df_orig['mag']    # @Tobias: which colum should we use?
df['NCCP'] = df_orig['NCCP']
df['NCCS'] = df_orig['NCCS']
df['NCTP'] = df_orig['NCTP']
df['NCTS'] = df_orig['NCTS']
df['RCC'] = df_orig['CC-RMS']
df['RCT'] = df_orig['CT-RMS']
df['CID'] = df_orig['CID']


# Store data in .csv file with ; as separator
df.to_csv(f'{out_path}hyporelocation_Valais_V0_preproc.csv', sep=';', index=False)