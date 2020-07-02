import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import datetime as dt
import inspect
from tensorflow.keras.utils import Sequence


def lineno():
    """Returns the current line number in our program."""
    return inspect.currentframe().f_back.f_lineno

# split a multivariate sequence into samples
def split_sequence(sequences, result, n_steps):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the dataset
		if end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x = sequences[i:end_ix, :]
		seq_y = result[end_ix-1, -1]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)

# Stacking the vectors into the correct input format
def reshape(vector):
    g = list()
    for i in range(len(vector)):
		# g.append(vector.loc[i])
        g.append(vector[i])
    x = np.array(g)
    vector = x.reshape((len(x), 1))
    return vector


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

def create_delays(df, name, time=20):
    for delay in np.arange(1,int(time)+1):
        df[name+'_%s' %delay] = df[name].shift(delay).astype('float32')

def create_average_delays(df, name, time=12):
    for delay in np.arange(10,int(time)*10+1,10):
        df[name+'_avg_%s' %delay] = df[name].shift(delay).rolling(10).mean().astype('float32')

def plotOptions(ax):
    """
    Defines the most general characteristics for the plots in
    this notebook.
    :label size for x and y axis
    :linewidth for the frame
    """
    plt.tick_params(axis='x', labelsize='16')
    plt.tick_params(axis='y', labelsize='16')
    for axis in ['top','bottom','left','right']:
      ax.spines[axis].set_linewidth(1.4)

class Generator(Sequence):
    """Generates data for Keras
    Sequence based data generator. Suitable for building data generator for training and prediction.
    """
    def __init__(self, features, results, to_fit=True, batch_size=32, shuffle=False):
        """Initialization
        :param list_IDs: list of all 'label' ids to use in the generator
        :param image_path: path to images location
        :param mask_path: path to masks location
        :param to_fit: True to return X and y, False to return X only
        :param batch_size: batch size at each iteration
        :param shuffle: True to shuffle label indexes after every epoch
        """
        self.features = features
        self.results = results
        self.to_fit = to_fit
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        return int(np.floor(len(self.features) / self.batch_size))


    def __getitem__(self, index):
        """Generate one batch of data
        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """

        # Generate data
        X = np.empty((self.batch_size, self.features.shape[1], self.features.shape[2]))
        X = self.features[index * self.batch_size:(index + 1) * self.batch_size]

        # Generate data

        if self.to_fit:
            y = np.empty((self.batch_size, 1))
            y = self.results[index * self.batch_size:(index + 1) * self.batch_size]
            return X, y
        else:
            return X


def get_results(real, predicted, date, window, threshold):
    ''' Compiles the magnitude, dBT into a pandas data frame and calculates prediction scores. '''

    date=date[-len(predicted):]
    date = date.reset_index(drop=True)
    pd.to_datetime(date, format = '%Y-%m-%d %H:%M:%S')
    d = pd.DataFrame({'real_B_H': real,
                        'predicted_B_H': predicted})
    d['real_dBT'] = d['real_B_H'].diff(-1)
    d['predicted_dBT'] = d['predicted_B_H'].diff(-1)
    d['date'] = date
    pod, pofd, pc, hss = classification_scores(d, window, threshold)
    d.set_index('date', inplace=True, drop=True)
    d.index = pd.to_datetime(d.index)
    return d, pod, pofd, pc, hss

def classification_scores(df, window, threshold):
    ''' calculates prediciton scores. a is correctly predicted hit, b is a false alarm, 
        c is a miss and d is a correcttly predicted non event. accepts a pandas dataframe'''

    a, b, c, d = 0,0,0,0
    for i in range(0, len(df), window):
        pred = df['predicted_dBT'][i:i+window]
        re = df['real_dBT'][i:i+window]
        if max(pred.max(),-pred.min()) > threshold and max(re.max(),-re.min()) > threshold:
            a =a+1
        elif max(pred.max(),-pred.min()) > threshold and max(re.max(),-re.min()) < threshold:
            b =b+1
        elif max(pred.max(),-pred.min()) < threshold and max(re.max(),-re.min()) > threshold:
            c =c+1
        elif max(pred.max(),-pred.min()) < threshold and max(re.max(),-re.min()) < threshold:
            d =d+1

    pod = a/(a+c)
    pofd = b/(b+d)
    pc = (a+d)/(a+b+c+d)
    hss = (2*((a*d)-(b*c)))/((a+c)*(c+d)+(a+b)*(b+d))

    return pod, pofd, pc, hss










