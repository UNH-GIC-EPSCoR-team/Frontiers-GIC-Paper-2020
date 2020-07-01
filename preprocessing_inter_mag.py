
import inspect
from gicfunctions import *
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

#dataDir = 'D:/Data/supermag/'
dataDir = '../../../data/supermag/baseline/'
plotDir = '../plots/'
omni_dir = '../../../data/omni/hro_1min/'

syear = 2011
smonth = 1
sday = 1
eyear = 2011
emonth = 12
eday = 31

start_time = str(pd.Timestamp(syear,smonth,sday))
start_time = start_time.replace(' ', '').replace('-', '').replace(':', '')
end_time = str(pd.Timestamp(eyear,emonth,eday,23,59,59))
end_time = end_time.replace(' ', '').replace('-', '').replace(':', '')


stations = ['OTT']
method = 'linear'
limit = 10

############################################################################################
####### Load and pre-process magnetometer data
############################################################################################

magFiles = glob.glob(dataDir+'%s/%s-*-supermag-baseline.csv' % (stations[0],stations[0]))
m = []

# Load original mag data for training years
# Convert original time to Pandas Datetime format, use datetime as DataFrame index
# and then fill in the missing date values (filling with NaN for now)
#entry = dataDir+'%s/%s-%s-supermag.csv' % (stations[0], stations[0], years[0])
for entry in magFiles:
    df = pd.read_csv(entry)
    df.drop('IAGA', axis=1, inplace=True)
    df['Date_UTC'] = pd.to_datetime(df['Date_UTC'])
    df.set_index('Date_UTC', inplace=True, drop=True)
    df = df.reindex(pd.date_range(start=dt.datetime(df.index[0].year, 1, 1), end=dt.datetime(df.index[0].year, 12, 31, 23, 59), freq='1 Min'), copy=True, fill_value=np.NaN)
    df['Date_UTC'] = df.index

    # Add magnitude and differential values
    df['MAGNITUDE'] = np.sqrt(df['N'] ** 2 + df['E'] ** 2 + df['Z'] ** 2)

    m.append(df)  

# Concatenate all DataFrames in a single DataFrame
magData = pd.concat(m, axis = 0, ignore_index=True)
magData.index = magData.Date_UTC
magData = magData[start_time:end_time]

#interpolating over missing data within a limit
magData['Z'] = magData.Z.interpolate(method=method)
magData['E'] = magData.E.interpolate(method=method)
magData['N'] = magData.N.interpolate(method=method)
magData['MAGNITUDE'] = magData.MAGNITUDE.interpolate(method=method)

# Calculating the time derivative of the horizontal component of the magnetic field.
magData['dBT'] = np.sqrt(((magData['N'].diff(-1))**2)+((magData['E'].diff(-1))**2))
# magData['dBT'] = magData['MAGNITUDE'].diff(-1)

magData['SZA'] = magData.SZA.interpolate(method=method)
magData['IGRF_DECL'] = magData.IGRF_DECL.interpolate(method=method)

magData['MLAT'] = magData.MLAT.interpolate(method=method)
magData['MLT'] = magData.MLT.interpolate(method=method)
# magData['MLT'] = magData.MLT.interpolate(method=method, limit=limit)

#to_drop = ['Date_UTC']
#magData.drop(to_drop, axis=1, inplace=True)

print(magData[-10:])

magData.to_csv(dataDir+'/%s/interpolated_supermag_%s-%s_nolimit.csv' % (stations[0], syear, eyear))


