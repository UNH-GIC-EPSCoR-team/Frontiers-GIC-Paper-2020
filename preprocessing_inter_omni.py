
import inspect

def lineno():
    """Returns the current line number in our program."""
    return inspect.currentframe().f_back.f_lineno


def break_dates(df, dateField, drop=False, errors="raise"):	
    """break_dates expands a column of df from a datetime64 to many columns containing
    the information from the date. This applies changes inplace.
    Parameters:
    -----------
    df: A pandas data frame. df gain several new columns.
    dateField: A string that is the name of the date column you wish to expand.
        If it is not a datetime64 series, it will be converted to one with pd.to_datetime.
    drop: If true then the original date column will be removed.
    
    Modified from FastAI software by Victor Pinto. 
    """
    field = df[dateField]
    field_dtype = field.dtype
    if isinstance(field_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
        field_dtype = np.datetime64

    if not np.issubdtype(field_dtype, np.datetime64):
        df[dateField] = field = pd.to_datetime(field, infer_datetime_format=True, errors=errors)
    
    attr = ['Year', 'Month', 'Day', 'Dayofyear', 'Hour', 'Minute']
    
    for n in attr: df[n] = getattr(field.dt, n.lower())
    if drop: df.drop(dateField, axis=1, inplace=True)
        
def omnicdf2dataframe(file):
    """
    Load a CDF File and convert it in a Pandas DataFrame.

    WARNING: This will not return the CDF Attributes, just the variables.
    WARNING: Only works for CDFs of the same array lenght (OMNI)
    """
    cdf = cdflib.CDF(file)
    cdfdict = {}

    for key in cdf.cdf_info()['zVariables']:
        cdfdict[key] = cdf[key]

    cdfdf = pd.DataFrame(cdfdict)

    if 'Epoch' in cdf.cdf_info()['zVariables']:
        cdfdf['Epoch'] = pd.to_datetime(cdflib.cdfepoch.encode(cdfdf['Epoch'].values))

    return cdfdf

def clean_omni(df):
    """    
    Remove filling numbers for missing data in OMNI data (1 min) and replace 
    them with np.nan values

    """

    # IMF
    df.loc[df['F'] >= 9999.99, 'F'] = np.nan
    df.loc[df['BX_GSE'] >= 9999.99, 'BX_GSE'] = np.nan
    df.loc[df['BY_GSE'] >= 9999.99, 'BY_GSE'] = np.nan
    df.loc[df['BZ_GSE'] >= 9999.99, 'BZ_GSE'] = np.nan
    df.loc[df['BY_GSM'] >= 9999.99, 'BY_GSM'] = np.nan
    df.loc[df['BZ_GSM'] >= 9999.99, 'BZ_GSM'] = np.nan
    
    # Speed
    df.loc[df['flow_speed'] >= 99999.9, 'flow_speed'] = np.nan
    df.loc[df['Vx'] >= 99999.9, 'Vx'] = np.nan
    df.loc[df['Vy'] >= 99999.9, 'Vy'] = np.nan
    df.loc[df['Vz'] >= 99999.9, 'Vz'] = np.nan
    
    # Particles
    df.loc[df['proton_density'] >= 999.99, 'proton_density'] = np.nan
    df.loc[df['T'] >= 9999999.0, 'T'] = np.nan
    df.loc[df['Pressure'] >= 99.99, 'Pressure'] = np.nan
    
    # Other
    df.loc[df['E'] >= 999.99, 'E'] = np.nan
    df.loc[df['Beta'] >= 999.99, 'Beta'] = np.nan
    df.loc[df['Mach_num'] >= 999.9, 'Mach_num'] = np.nan
    
    # Indices
    df.loc[df['AE_INDEX'] >= 99999, 'AE_INDEX'] = np.nan
    df.loc[df['AL_INDEX'] >= 99999, 'AL_INDEX'] = np.nan
    df.loc[df['AU_INDEX'] >= 99999, 'AU_INDEX'] = np.nan
    df.loc[df['SYM_D'] >= 99999, 'SYM_D'] = np.nan
    df.loc[df['SYM_H'] >= 99999, 'ASY_D'] = np.nan
    df.loc[df['ASY_H'] >= 99999, 'ASY_H'] = np.nan
    df.loc[df['PC_N_INDEX'] >= 999, 'PC_N_INDEX'] = np.nan
    
    return(df)

import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import datetime as dt
import os
os.environ["CDF_LIB"] = "~/lib"
import cdflib

unix_time = dt.date(1971,1,1).toordinal()*24*60*60
plt.rcParams['figure.figsize'] = [15, 5]

syear = 2011
eyear = 2011

#dataDir = 'D:/Data/supermag/'
# dataDir = '../Data_files/Supermag/OTT/'
# plotDir = '../Data_files/Supermag/OTT/plots'
omni_dir = '../../../data/omni/hro_1min/'

start_time = str(pd.Timestamp(syear,1,1))
start_time = start_time.replace(' ', '').replace('-', '').replace(':', '')
end_time = str(pd.Timestamp(eyear,12,31,23,59,59))
end_time = end_time.replace(' ', '').replace('-', '').replace(':', '')

method = 'linear'
limit=10
stations = ['OTT']

############################################################################################
####### Load and pre-process solar wind data
############################################################################################

# omniFiles = (omni_dir+'omni_hro_1min_19990401_v01.cdf')
omniFiles = glob.glob(omni_dir+'*/*.cdf', recursive=True)

o = []
for fil in sorted(omniFiles):
    cdf = omnicdf2dataframe(fil)
    o.append(cdf)
    
omni_start_time = str(pd.Timestamp(syear,1,1))
omni_start_time = omni_start_time.replace(' ', '').replace('-', '').replace(':', '')
omni_end_time = str(pd.Timestamp(eyear,12,31,23,59,59))
omni_end_time = omni_end_time.replace(' ', '').replace('-', '').replace(':', '')

omniData = pd.concat(o, axis = 0, ignore_index = True)
omniData.index = omniData.Epoch
omniData = omniData[omni_start_time:omni_end_time]

to_drop = ['IMF', 'PLS', 'IMF_PTS', 'PLS_PTS', 'percent_interp', 
           'Timeshift', 'RMS_Timeshift', 'RMS_phase', 'Time_btwn_obs',
           'RMS_SD_B', 'RMS_SD_fld_vec', 'x', 'y', 
           'z', 'BSN_x', 'BSN_y', 'BSN_z']

omniData = omniData.drop(to_drop, axis=1)
clean_omni(omniData)

omniData.rename(columns={'E': 'E_Field'}, inplace=True)
omniData.rename(columns={'F': 'B_Total'}, inplace=True)
# omniData.describe().T

# Process the data. Right now interpolation on all missing data. May change later

omniData['Vx'] = omniData.Vx.interpolate(method=method)
omniData['Vy'] = omniData.Vy.interpolate(method=method)
omniData['Vz'] = omniData.Vz.interpolate(method=method)

omniData['proton_density'] = omniData.proton_density.interpolate(method=method)

omniData['BZ_GSM'] = omniData.BZ_GSM.interpolate(method=method)
omniData['BY_GSM'] = omniData.BY_GSM.interpolate(method=method)

omniData['BX_GSE'] = omniData.BX_GSE.interpolate(method=method)
omniData['BY_GSE'] = omniData.BY_GSE.interpolate(method=method)
omniData['BZ_GSE'] = omniData.BZ_GSE.interpolate(method=method)

omniData['B_Total'] = omniData.B_Total.interpolate(method=method)

# We want to ideally calculate this variables again and not interpolate them, but at this moment we are using interpolation to finish in time

omniData['flow_speed'] = omniData.flow_speed.interpolate(method=method)
omniData['T'] = omniData['T'].interpolate(method=method)
omniData['Pressure'] = omniData.Pressure.interpolate(method=method)
omniData['E_Field'] = omniData.E_Field.interpolate(method=method)
omniData['Beta'] = omniData.Beta.interpolate(method=method)
omniData['Mach_num'] = omniData.Mach_num.interpolate(method=method)
omniData['Mgs_mach_num'] = omniData.Mgs_mach_num.interpolate(method=method)
# omniData['Mgs_mach_num'] = omniData.Mgs_mach_num.interpolate(method=method, limit=limit)

print(omniData[-5:])

# allData = pd.concat([omniData, magData], axis = 1, ignore_index = False)
# allData.columns.T
# Maybe drop the data early in the processing
to_drop = ['Epoch', 'YR', 'Day', 'HR', 'Minute']
omniData.drop(to_drop, axis=1, inplace=True)

omniData.to_csv(omni_dir+'test_interpolated_omni_%s-%s_nolimit.csv' % (syear, eyear))


