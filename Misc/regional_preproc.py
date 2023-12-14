### Pre-processing of regional relocation data

import pandas as pd
from pyproj import Transformer
import numpy as np

in_file = '/Users/sandro/Library/CloudStorage/Dropbox/PhD/Data/Data_SED/01_Valais/V0/01_OriginalData/hyporelocation_V0_Valais.xlsx'
out_path = '/Users/sandro/Library/CloudStorage/Dropbox/PhD/Data/Data_SED/01_Valais/V0/'
# Read in the .xlsx file
df_orig = pd.read_excel(in_file)
print('Number of events in original dataframe: ', len(df_orig))
print(df_orig.head())
print(df_orig['PreMet'].unique())

#######################
# Filtering df_orig
# Only select data if PreMet == 'SVD' or 'LSQ'
df_orig = df_orig[(df_orig['PreMet'] == 'LSQ') | (df_orig['PreMet'] == 'SVD')]

# Only select data with CAT-ID 'DDR-SW0' (Valais)
df_orig = df_orig[df_orig['CAT-ID'] == 'DDR-SW0']

# # Only keep tectonic events (labelled with 'T')
# df_orig = df_orig[df_orig['T'] == 'T']

# # Remove data prior to certain year
# df_orig = df_orig[df_orig['YYYY'] >= 1900]

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

# Tranform coordinates from lat/lon (WGS84) to CH1903+ (EPSG:2056) using pyproj
transformer = Transformer.from_crs("epsg:4326", "epsg:2056")
x, y = transformer.transform(df_orig['Pref-lat'].values, df_orig['Pref-lon'].values)

# TODO: check with Tobias if right columns are used here!!!
# Fill the new dataframe with the data from the original dataframe
df['ID'] = df_orig['#DD-ID']
df['LAT'] = df_orig['Pref-lat']
df['LON'] = df_orig['Pref-lon']
df['DEPTH'] = df_orig['Pref-dep']
df['X'] = x     # @Tobias: which colum should we use?
df['Y'] = y     # @Tobias: which colum should we use?
df['Z'] = df_orig['Pref-dep'] * 1000    # @Tobias: which colum should we use?
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
df.to_csv(f'{out_path}hyporelocation_V0_Valais_preproc.csv', sep=';', index=False)



###################################################################################
# Pre-processing pipeline for focal mechanisms, ignoring first line of .txt file
df_foc_orig = pd.read_excel('/Users/sandro/Library/CloudStorage/Dropbox/PhD/Data/Data_SED/01_Valais/V0/01_OriginalData/focals_V0_Valais.xlsx')

# Create empty dataframe with pre-defined columns with the necessery input structure
df_foc = pd.DataFrame(columns=['Yr', 'Mo', 'Dy', 'Hr:Mi', 'Lat', 'Lon', 'Z', 'Mag',
                               'A', 'Strike1', 'Dip1', 'Rake1', 'Strike2', 'Dip2', 'Rake2',
                               'Pazim', 'Pdip', 'Tazim', 'Tdip', 'Q', 'Type', 'Loc'])

# Tranform coordinates from lat/lon (WGS84) to CH1903+ (EPSG:2056) using pyproj
transformer = Transformer.from_crs("epsg:4326", "epsg:2056")
x, y = transformer.transform(df_foc_orig['Lat'].values, df_foc_orig['Lon'].values)

# TODO: check with Tobias if right columns are used here!!!
# Fill the new dataframe with the data from the original dataframe
# Extract year, month and day from df_foc_orig['YYYY/MM/DD']
year = []
month = []
day = []
for i in range(len(df_foc_orig['YYYY/MM/DD'])):
    year.append(df_foc_orig['YYYY/MM/DD'][i].year)
    month.append(df_foc_orig['YYYY/MM/DD'][i].month)
    day.append(df_foc_orig['YYYY/MM/DD'][i].day)

df_foc['Yr'] = year
df_foc['Mo'] = month
df_foc['Dy'] = day
df_foc['Hr:Mi'] = df_foc_orig['HH/MI/SS.S']
df_foc['Lat'] = df_foc_orig['Lat']
df_foc['Lon'] = df_foc_orig['Lon']
df_foc['Z'] = df_foc_orig['Dep']
df_foc['Mag'] = df_foc_orig['Mag']
df_foc['A'] = df_foc_orig['AP']
df_foc['Strike1'] = df_foc_orig['S1']
df_foc['Dip1'] = df_foc_orig['D1']
df_foc['Rake1'] = df_foc_orig['R1']
df_foc['Strike2'] = df_foc_orig['S2']
df_foc['Dip2'] = df_foc_orig['D2']
df_foc['Rake2'] = df_foc_orig['R2']
df_foc['Pazim'] = df_foc_orig['Paz']
df_foc['Pdip'] = df_foc_orig['Ppl']
df_foc['Tazim'] = df_foc_orig['Taz']
df_foc['Tdip'] = df_foc_orig['Tpl']
df_foc['Q'] = df_foc_orig['FMQ']
df_foc['Type'] = df_foc_orig['FMT']
df_foc['Loc'] = np.nan

# Store data in .csv file with ; as separator
df_foc.to_csv(f'{out_path}focals_V0_Valais_preproc.csv', sep=';', index=False)

