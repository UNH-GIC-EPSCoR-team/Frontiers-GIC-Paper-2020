import dask.dataframe as dd
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import datetime as dt
import inspect
from LSTMfunctions import *
import time
import matplotlib.dates as mdates

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Activation, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import Sequence

unix_time = dt.date(1971,1,1).toordinal()*24*60*60

# Data directories
dataDir = '../../../data/mikes_files/supermag/OTT/'
plotDir = '../plots/'
omniDir = '../../../data/mikes_files/omni/'

time_steps = 12

# Getting data from supermag and omni and then combining into one Dataframe
# Training Data
magData_train = pd.read_csv(dataDir+'interpolated_supermag_1995-2010_nolimit.csv')
# magData_train = pd.read_csv(dataDir+'interpolated_supermag_2005-2007_limit10.csv')
pd.to_datetime(magData_train['Date_UTC'], format = '%Y-%m-%d %H:%M:%S')
magData_train.set_index('Date_UTC', inplace=True, drop=False)
magData_train.index = pd.to_datetime(magData_train.index)
omniData_train = pd.read_csv(omniDir+'test_interpolated_omni_1995-2010_nolimit.csv')
pd.to_datetime(omniData_train['Epoch'], format = '%Y-%m-%d %H:%M:%S')
omniData_train.set_index('Epoch', inplace=True, drop=True)
omniData_train.index = pd.to_datetime(omniData_train.index)

magData_train['sinMLT'] = np.sin(magData_train.MLT * 2 * np.pi * 15 / 360)
magData_train['cosMLT'] = np.cos(magData_train.MLT * 2 * np.pi * 15 / 360)

magData_train.drop(labels=['MLT','IGRF_DECL', 'SZA', 'Date_UTC.1'], axis=1)

omni_drop = ['B_Total','PC_N_INDEX', 
               'Beta', 'Mach_num', 'Mgs_mach_num', 'AE_INDEX', 'AL_INDEX', 
               'AU_INDEX', 'SYM_D', 'SYM_H', 'ASY_D', 'ASY_H']

omniData_train = omniData_train.drop(omni_drop, axis=1)

# Creating the rolling averages for the omni data
create_average_delays(omniData_train, omniData_train.Vx.name, time=time_steps)
create_average_delays(omniData_train, omniData_train.Vy.name, time=time_steps)
create_average_delays(omniData_train, omniData_train.Vz.name, time=time_steps)
create_average_delays(omniData_train, omniData_train.proton_density.name, time=time_steps)
create_average_delays(omniData_train, omniData_train.BY_GSM.name, time=time_steps)
create_average_delays(omniData_train, omniData_train.BZ_GSM.name, time=time_steps)
create_average_delays(omniData_train, omniData_train.BX_GSE.name, time=time_steps)
create_average_delays(omniData_train, omniData_train.E_Field.name, time=time_steps)

# Concatinating data
trainData = pd.concat([magData_train, omniData_train], axis = 1, ignore_index=False)
del magData_train, omniData_train

# Dropping NaN values
print('Dropping NaN')
trainData = trainData.dropna()

# Calculating the magnitude of B. May change this to be done after the model runs if that gives better results
x = np.sqrt(trainData['N'] ** 2 + trainData['E'] ** 2)

# Assigning shorter variables to work with
X = trainData.drop(labels=['N', 'E', 'Z', 'MAGNITUDE', 'dBT','Date_UTC.1'], axis=1)
del trainData

print('Train-testing split')

X_train, X_val, y_train, y_val = train_test_split(X,x,test_size=0.3, shuffle=False)

del X, x

X_train = X_train.drop(labels='Date_UTC', axis=1)
X_val = X_val.drop(labels='Date_UTC', axis=1)

X_train.reset_index(inplace = True, drop = True)
X_val.reset_index(inplace = True, drop = True)
y_train.reset_index(inplace = True, drop = True)
y_val.reset_index(inplace = True, drop = True)

# scaling the training data
scaler = MinMaxScaler()
print('Fitting X train')
scaler.fit(X_train)
print('Scaling X train')
X_train = scaler.transform(X_train)
print('Scaling X val')
X_val = scaler.transform(X_val)

# Reshaping the matricies
print('Reshaping train data')
y_train = reshape(y_train)
print('Reshaping val data')
y_val = reshape(y_val)

# Splitting up the data into the 12 minute prior sequences and with delay columns for variables
n_steps = 12
print('spliting model training data')
X_train, y_train = split_sequence(X_train, y_train, n_steps)
print('spliting model validation data')
X_val, y_val = split_sequence(X_val, y_val, n_steps)

n_features = X_train.shape[2]

print('Shape of training data after split:'+str(X_train.shape))

train_gen = Generator(X_train, y_train, batch_size=2048)
val_gen = Generator(X_val, y_val, batch_size=2048)

del X_train, y_train

trun = []
rmse_2015, r2_2015, expv_2015, mae_2015 = [],[],[],[]
rmse_2011, r2_2011, expv_2011, mae_2011 = [],[],[],[]
rmse_val, r2_val, expv_val, mae_val = [],[],[],[]

# defining the one layer-LSTM model and one dense layer model
srunTime = time.time()
model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)

# fit first model
model.fit(train_gen, validation_data=val_gen, verbose=1, shuffle=False, epochs=500, callbacks=[early_stop])
model.save('../models/LSTM_w_avgs.h5')
losses = pd.DataFrame(model.history.history)
losses.plot()

plt.figure(1)
# plt.ylim(0, 200)
plt.title('LSTM Losses')
plt.xlabel('Epochs')
plt.legend()
plt.savefig(plotDir+'LSTM_avg_losses.png')
# model = load_model('../models/LSTM_w_avgs.h5')

del train_gen, val_gen

'''Loading in testing data'''

# Testing Data
magData_2015 = pd.read_csv(dataDir+'interpolated_supermag_2015-2015_nolimit.csv')
magData_2015.set_index('Date_UTC', inplace=True, drop=False)
omniData_2015 = pd.read_csv(omniDir+'test_interpolated_omni_2015-2015_nolimit.csv')
omniData_2015.set_index('Epoch', inplace=True, drop=True)

# Validation Data
magData_2011 = pd.read_csv(dataDir+'interpolated_supermag_2011-2011_nolimit.csv')
magData_2011.set_index('Date_UTC', inplace=True, drop=False)
omniData_2011 = pd.read_csv(omniDir+'test_interpolated_omni_2011-2011_nolimit.csv')
omniData_2011.set_index('Epoch', inplace=True, drop=True)

magData_2015['sinMLT'] = np.sin(magData_2015.MLT * 2 * np.pi * 15 / 360)
magData_2015['cosMLT'] = np.cos(magData_2015.MLT * 2 * np.pi * 15 / 360)
magData_2011['sinMLT'] = np.sin(magData_2011.MLT * 2 * np.pi * 15 / 360)
magData_2011['cosMLT'] = np.cos(magData_2011.MLT * 2 * np.pi * 15 / 360)

magData_2015.drop(labels=['MLT','IGRF_DECL', 'SZA', 'Date_UTC.1'], axis=1)
magData_2011.drop(labels=['MLT','IGRF_DECL', 'SZA', 'Date_UTC.1'], axis=1)

omniData_2015 = omniData_2015.drop(omni_drop, axis=1)
omniData_2011 = omniData_2011.drop(omni_drop, axis=1)

# creating rolling averages for test data
create_average_delays(omniData_2015, omniData_2015.Vx.name, time=time_steps)
create_average_delays(omniData_2015, omniData_2015.Vy.name, time=time_steps)
create_average_delays(omniData_2015, omniData_2015.Vz.name, time=time_steps)
create_average_delays(omniData_2015, omniData_2015.proton_density.name, time=time_steps)
create_average_delays(omniData_2015, omniData_2015.BY_GSM.name, time=time_steps)
create_average_delays(omniData_2015, omniData_2015.BZ_GSM.name, time=time_steps)
create_average_delays(omniData_2015, omniData_2015.BX_GSE.name, time=time_steps)
create_average_delays(omniData_2015, omniData_2015.E_Field.name, time=time_steps)

create_average_delays(omniData_2011, omniData_2011.Vx.name, time=time_steps)
create_average_delays(omniData_2011, omniData_2011.Vy.name, time=time_steps)
create_average_delays(omniData_2011, omniData_2011.Vz.name, time=time_steps)
create_average_delays(omniData_2011, omniData_2011.proton_density.name, time=time_steps)
create_average_delays(omniData_2011, omniData_2011.BY_GSM.name, time=time_steps)
create_average_delays(omniData_2011, omniData_2011.BZ_GSM.name, time=time_steps)
create_average_delays(omniData_2011, omniData_2011.BX_GSE.name, time=time_steps)
create_average_delays(omniData_2011, omniData_2011.E_Field.name, time=time_steps)

# Concatinating the data and dropping NaN
Data2015 = pd.concat([magData_2015, omniData_2015], axis = 1, ignore_index=False)
Data2011 = pd.concat([magData_2011, omniData_2011], axis = 1, ignore_index=False)
train2015 = Data2015.dropna()
train2011 = Data2011.dropna()

# Defining the real magnitude values
test2015 = np.sqrt(train2015['N'] ** 2 + train2015['E'] ** 2)
test2011 = np.sqrt(train2011['N'] ** 2 + train2011['E'] ** 2)

train2015 = train2015.drop(labels=['N', 'E', 'Z', 'MAGNITUDE', 'dBT', 'Date_UTC.1'], axis=1)
train2011 = train2011.drop(labels=['N', 'E', 'Z', 'MAGNITUDE', 'dBT', 'Date_UTC.1'], axis=1)
date_2015 = train2015['Date_UTC']
date_2011 = train2011['Date_UTC']

date_2015.reset_index(inplace = True, drop = True)
date_2011.reset_index(inplace = True, drop = True)

train2015 = train2015.drop(labels='Date_UTC', axis=1)
train2011 = train2011.drop(labels='Date_UTC', axis=1)

train2015.reset_index(inplace = True, drop = True)
test2015.reset_index(inplace = True, drop = True)
train2011.reset_index(inplace = True, drop = True)
test2011.reset_index(inplace = True, drop = True)

print('Scaling 2015 training data')
train2015 = scaler.transform(train2015)
print('Scaling 2011 training data')
train2011 = scaler.transform(train2011)

print('Reshaping 2015 test data')
test2015 = reshape(test2015)
print('Reshaping 2011 test data')
test2011 = reshape(test2011)

print('splitting 2015 data')
train2015, test2015 = split_sequence(train2015, test2015, n_steps)
print('splitting 2011 data')
train2011, test2011 = split_sequence(train2011, test2011, n_steps)

yhat_val = model.predict(X_val, verbose=1)
yhat_2015 = model.predict(train2015, verbose=1)
yhat_2011 = model.predict(train2011, verbose=1)

rmse_val.append(np.sqrt(mean_squared_error(y_val,yhat_val)))
mae_val.append(mean_absolute_error(y_val,yhat_val))
expv_val.append(explained_variance_score(y_val,yhat_val))
r2_val.append(r2_score(y_val,yhat_val))

rmse_2015.append(np.sqrt(mean_squared_error(test2015,yhat_2015)))
mae_2015.append(mean_absolute_error(test2015,yhat_2015))
expv_2015.append(explained_variance_score(test2015,yhat_2015))
r2_2015.append(r2_score(test2015,yhat_2015))

rmse_2011.append(np.sqrt(mean_squared_error(test2011,yhat_2011)))
mae_2011.append(mean_absolute_error(test2011,yhat_2011))
expv_2011.append(explained_variance_score(test2011,yhat_2011))
r2_2011.append(r2_score(test2011,yhat_2011))

erunTime = time.time()
trunTime = erunTime - srunTime
trun.append(trunTime)

print("Finished iteration. Time duration: %s" % trunTime)

scores = pd.DataFrame({'trun': trun,
					   'rmse_val': rmse_val,
					   'mae_val':mae_val,
					   'expv_val': expv_val,
					   'r2_val':r2_val,
					   'rmse_2015': rmse_2015,
					   'mae_2015':mae_2015,
					   'expv_2015': expv_2015,
					   'r2_2015':r2_2015,
					   'rmse_2011': rmse_2011,
					   'mae_2011':mae_2011,
					   'expv_2011': expv_2011,
					   'r2_2011':r2_2011})

scores.reset_index().to_csv(plotDir+'LSTM_avg_scores.csv')

# Making everything nice for plotting
yhat_val = pd.Series(yhat_val.reshape(len(yhat_val),))
y_val = pd.Series(y_val.reshape(len(y_val),))
yhat_2015 = pd.Series(yhat_2015.reshape(len(yhat_2015),))
test2015 = pd.Series(test2015.reshape(len(test2015),))
yhat_2011 = pd.Series(yhat_2011.reshape(len(yhat_2011),))
test2011 = pd.Series(test2011.reshape(len(test2011),))

predict2015, pod2015, pofd2015, pc2015, hss2015 = get_results(test2015, yhat_2015, date_2015, window=20, threshold=18)
predict2011, pod2011, pofd2011, pc2011, hss2011 = get_results(test2011, yhat_2011, date_2011, window=20, threshold=18)

predictions = pd.DataFrame({'POD_2015': [pod2015],
                       'POFD_2015': [pofd2015],
                       'PC_2015':[pc2015],
                       'HSS_2015': [hss2015],
                       'POD_2011':[pod2011],
                       'POFD_2011': [pofd2011],
                       'PC_2011':[pc2011],
                       'HSS_2011': [hss2011]})

predictions.reset_index().to_csv(plotDir+'LSTM_prediction_scores.csv')

'''TIME FOR ALL THE PLOTTING!!!!!'''

# Getting R value
R = []
R.append(np.sqrt(r2_score(y_val,yhat_val)))
R.append(np.sqrt(r2_score(test2015,yhat_2015)))
R.append(np.sqrt(r2_score(test2011,yhat_2011)))

# Individual heatmap plot
fig = plt.figure(figsize=[15,10])
cmap = 'plasma'
x = np.linspace(-5,5)
fontsize = 16

ax = fig.add_subplot(111)
plotOptions(ax)
plt.title('Validation LSTM', fontsize=fontsize)
plt.ylabel('$ log_{10}(B_H)$ Predicted', fontsize=fontsize)
plt.xlabel('$ log_{10}(B_H)$ Real', fontsize=fontsize)
plt.hist2d(np.log10(y_val), np.log10(np.ravel(yhat_val)), bins=(250, 250), 
           range=[[-0.5, 3.5], [-0.5, 3.5]], cmap=cmap, cmin=1)
plt.colorbar().set_label(label='Points per bin', size=fontsize)
plt.plot(x,x, 'k-', label='R = %.2f' % R[0])
plt.legend(loc='upper left', fontsize=fontsize, handlelength=0)
plt.tight_layout()
plt.savefig(plotDir+'validation_heatmap_avg.png')

fig = plt.figure(figsize=[15,10])

ax = fig.add_subplot(111)
plotOptions(ax)
plt.title('Test LSTM - 2015', fontsize=fontsize)
plt.ylabel('$ log_{10}(B_H)$ Predicted', fontsize=fontsize)
plt.xlabel('$ log_{10}(B_H)$ Real', fontsize=fontsize)
plt.hist2d(np.log10(test2015), np.log10(np.ravel(yhat_2015)), bins=(250, 250), 
           range=[[-0.5, 3.5], [-0.5, 3.5]], cmap=cmap, cmin=1)
plt.colorbar().set_label(label='Points per bin', size=fontsize)
plt.plot(x,x, 'k-', label='R = %.2f' % R[1])
plt.legend(loc='upper left', fontsize=fontsize, handlelength=0)
plt.tight_layout()
plt.savefig(plotDir+'2015_heatmap_avg.png')

fig = plt.figure(figsize=[15,10])

ax = fig.add_subplot(111)
plotOptions(ax)
plt.title('Test LSTM - 2011', fontsize=fontsize)
plt.ylabel('$ log_{10}(B_H)$ Predicted', fontsize=fontsize)
plt.xlabel('$ log_{10}(B_H)$ Real', fontsize=fontsize)
plt.hist2d(np.log10(test2011), np.log10(np.ravel(yhat_2011)), bins=(250, 250), 
           range=[[-0.5, 3.5], [-0.5, 3.5]], cmap=cmap, cmin=1)
plt.colorbar().set_label(label='Points per bin', size=fontsize)
plt.plot(x,x, 'k-', label='R = %.2f' % R[2])
plt.legend(loc='upper left', fontsize=fontsize, handlelength=0)
plt.tight_layout()
plt.savefig(plotDir+'2011_heatmap_avg.png')


# Combimed heatmap plots
fig = plt.figure(figsize=[17,9])
cmap = 'plasma'
# x = np.linspace(-2, 10000, 1)
x = np.linspace(-5,5)

ax1 = fig.add_subplot(131)
plotOptions(ax)
plt.title('Validation Data - LSTM', fontsize=16)
plt.ylabel('$ log_{10}(B_H)$ Predicted', fontsize=fontsize)
plt.xlabel('$ log_{10}(B_H)$ Real', fontsize=fontsize)
plt.hist2d(np.log10(y_val), np.log10(np.ravel(yhat_val)), bins=(250, 250), 
           range=[[-0.5, 3.5], [-0.5, 3.5]], cmap=cmap, cmin=1)
plt.plot(x,x, 'k-', label='R = %.2f' % R[0])
plt.legend(loc='upper left', fontsize=fontsize, handlelength=0)
plt.colorbar()

ax2 = fig.add_subplot(132)
plotOptions(ax)
plt.title('Test LSTM - 2011', fontsize=fontsize)
plt.xlabel('$ log_{10}(B_H)$ Real', fontsize=fontsize)
plt.hist2d(np.log10(test2015), np.log10(np.ravel(yhat_2015)), bins=(250, 250), 
           range=[[-0.5, 3.5], [-0.5, 3.5]], cmap=cmap, cmin=1)
plt.plot(x,x, 'k-', label='R = %.2f' % R[1])
plt.legend(loc='upper left', fontsize=fontsize, handlelength=0)
plt.colorbar()

ax3 = fig.add_subplot(133)
plotOptions(ax)
plt.title('Test LSTM - 2015', fontsize=fontsize)
plt.xlabel('$ log_{10}(B_H)$ Real', fontsize=fontsize)
plt.hist2d(np.log10(test2011), np.log10(np.ravel(yhat_2011)), bins=(250, 250), 
           range=[[-0.5, 3.5], [-0.5, 3.5]], cmap=cmap, cmin=1)
plt.colorbar().set_label(label='Points per bin', size=fontsize)
plt.plot(x,x, 'k-', label='R = %.2f' % R[2])
plt.legend(loc='upper left', fontsize=fontsize, handlelength=0)

plt.savefig(plotDir+'histogram2d-real-predicted_avg.png')


# B_H and dBT combined plots
def applyThisPlotStyle(panel):
    panels = ['OTT $B_H$ (nT)', '$dB_H/dt$ (nT/min)']
    plt.xlabel('')
    plt.ylabel(panels[panel],fontsize='20')

# 2015 plots
stime = '2015-03-17 00:00'
etime = '2015-03-18 12:00'
fig = plt.figure(figsize=(21,21))
plt.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.85, hspace=0.03)

ax = fig.add_subplot(211)   
applyThisPlotStyle(0)
plotOptions(ax)
plt.plot(predict2015[stime:etime].real_B_H , 'b-')
plt.plot(predict2015[stime:etime].predicted_B_H, 'r-', label='Prediction')
ax.set_xticklabels('')
# plt.ylim(0, 450)
# plt.xlim(predict2015[stime:etime].index[0], predict2015[stime:etime].index[-1])
plt.legend(fontsize='14')

ax = fig.add_subplot(212)   
applyThisPlotStyle(1)
plotOptions(ax)
plt.plot(predict2015[stime:etime].real_dBT , 'b-')
plt.plot(predict2015[stime:etime].predicted_dBT, 'r-', label='Prediction')
ax.set_xticklabels('')
# plt.ylim(-150, 120)
# plt.xlim(predict2015[stime:etime].index[0], predict2015[stime:etime].index[-1])
plt.legend(fontsize='14')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n %H:%M'))

plt.savefig(plotDir+'time-series-2015-storm_avg.png')

# 2011 plotsd
stime = '2011-08-05 12:00'
etime = '2011-08-06 12:00'
fig = plt.figure(figsize=(21,21))
plt.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.85, hspace=0.03)

ax = fig.add_subplot(211)   
applyThisPlotStyle(0)
plotOptions(ax)
plt.plot(predict2011[stime:etime].real_B_H , 'b-')
plt.plot(predict2011[stime:etime].predicted_B_H, 'r-', label='Prediction')
ax.set_xticklabels('')
plt.ylim(0, 450)
# plt.xlim(predict2015[stime:etime].index[0], predict2015[stime:etime].index[-1])
plt.legend(fontsize='14')

ax = fig.add_subplot(212)   
applyThisPlotStyle(1)
plotOptions(ax)
plt.plot(predict2011[stime:etime].real_dBT , 'b-')
plt.plot(predict2011[stime:etime].predicted_dBT, 'r-', label='Prediction')
ax.set_xticklabels('')
plt.ylim(-150, 120)
# plt.xlim(predict2015[stime:etime].index[0], predict2015[stime:etime].index[-1])
plt.legend(fontsize='14')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n %H:%M'))

plt.savefig(plotDir+'time-series-2011-storm_avg.png')


print('It ran! Good job!')













