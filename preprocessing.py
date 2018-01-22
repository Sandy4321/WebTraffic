# coding: utf-8

import pandas as pd
import numpy as np

import re
import os
import pickle
import numba

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

DATA_DIR = 'data'
FILE_NAME = 'train_1'
CSV_PATH = os.path.join(DATA_DIR, FILE_NAME + '.csv')
PKL_PATH = os.path.join(DATA_DIR, FILE_NAME + '.pkl')


def load_data() -> pd.DataFrame:
    '''
    Loads data from path. If there is a cached version loads it instead.
    '''
    if os.path.exists(PKL_PATH):
        print('Loading pickle...')
        df = pd.read_pickle(PKL_PATH)
        print('Done!')
        return df
    else:
        print('Loading csv...')
        df = pd.read_csv(CSV_PATH)
        df.to_pickle(PKL_PATH)
        print('Done!')
        return df


def read_all() -> pd.DataFrame:
    '''
    Loads data, sets index for the df and makes columns a date type.
    Also pickles for speed increase
    '''
    
    treated_pkl = os.path.join(DATA_DIR, 'treated.pkl')
    if os.path.exists(treated_pkl):
        df = pd.read_pickle(treated_pkl)
    else:
        df = load_data()
        df.set_index('Page', inplace=True)
        df.sort_index(inplace=True)
        df.columns = df.columns.astype('M8[D]')
        print('Pickling treated data...')
        df.to_pickle(treated_pkl)
        print('Done!')
    return df


def read_interval(start, end) -> pd.DataFrame:
    '''
    Returns dataframe within specified values: ts[start:end]
    '''
    df = read_all()
    if start and end:
        return df.loc[:, start:end]
    elif end:
        return df.loc[:, :end]
    else:
        return df


# 16 mins with dataframe access <br>
# 6 segs with numpy arrays <br>
# 289ms with numba <br>

def get_clean_data(threshold, start=None, end=None):
    '''
    Loads data, setting Page as index, and columns as datetime dtypes.
    Removes series that don't comply to minimum threshold of nan to value ratio
    Returns normalized series (log1p), indexes of previously nan values, start and end indexes
    '''
    df = read_interval(start, end)
    start, end = calculate_start_end(df.values)
    bool_mask = ~(((end - start) / df.shape[1]) < threshold)
    df = df[bool_mask]
    
    nan_values = pd.isnull(df)
    return np.log1p(df.fillna(0)), nan_values, start, end


def standard_scale(arr: np.ndarray):
    '''
    Normalize data (x - mean)/std
    '''
    return (arr - arr.mean()) / np.std(arr)


@numba.jit(nopython=True)
def autocorrelation(series: np.ndarray, days):
    '''
    Calculates autocorrelation for series according to Box Jenkins
    '''
    ts = series[days:]
    ts_lag = series[:-days]
    dts = ts - np.mean(ts)
   
    dts_lag = ts_lag - np.mean(ts_lag)
    
    
    #denominator = np.sum(np.square(dts))
    denominator = np.sqrt(np.sum(dts * dts)) + np.sqrt(np.sum(dts_lag * dts_lag))
    nominator = np.sum(dts * dts_lag)
    if denominator == 0 or np.isnan(denominator):
        autocorr = 0
    else:
        autocorr = nominator / denominator
    return autocorr


@numba.jit(nopython=True)
def batch_autocorrelation(tseries, lag, starts, ends, threshold):
    '''
    Calculates autocorrelation for lots of series.
    Checks if we are calculating a meaningful result for autocorrelation
    If the length of the series is less than the threshold (len/lag),
    autocorrelation is none.
    '''
    rows, columns = tseries.shape[0], tseries.shape[1]
    corr = np.full(rows, np.nan, dtype=np.float64)
    
    for i in range(rows):
        # check if series complies to threshold
        start = starts[i]
        end = min(ends[i], columns)
        ratio = (end - start) / lag
        if ratio > threshold:
            # calculate autocorrelation
            entry = tseries[i]
            corr[i] = autocorrelation(entry[start:end], lag)
        else:
            # autocorr is nan
            continue
    return corr

def get_autocorr(tseries: np.ndarray, lags: list, starts, ends, threshold, normalize=True):
    '''
    Gets autocorrelations for each specified lag in lags
    '''
    
    corr = [batch_autocorrelation(tseries, lag, starts, ends, threshold)
           for lag in lags]
        
    for i in range(len(lags)):
        ratio = (corr[i].shape[0] - np.sum(np.isnan(corr[i])) ) / corr[i].shape[0]
        nan_percent = 1 - ratio
        print("For lag: %i nan percent is %.3f" % (lags[i], nan_percent))
        
    if normalize:
        corr = [standard_scale(np.nan_to_num(batch)) for batch in corr]
    
    return corr



pat = re.compile(
    '(.+)_([a-z]{2}\.)?((?:wikipedia\.org)|(?:commons\.wikimedia\.org)|(?:www\.wikimedia\.org)|(?:www\.mediawiki\.org))_([a-z-]+?)_([a-z-]+)$'
)

def extract_from_url(urls: np.ndarray) -> pd.DataFrame:
    '''
    receives pandas dataframe column or series
    returns a pandas dataframe with all the extracted features
    '''
    
    if isinstance(urls, pd.Series):
         urls = urls.values
    
    accesses = np.full_like(urls, np.NaN)
    agents = np.full_like(urls, np.NaN)
    sites = np.full_like(urls, np.NaN)
    countries = np.full_like(urls, np.NaN)
    titles = np.full_like(urls, np.nan)
    
    for i in range(len(urls)):
        url = urls[i]
        match = pat.fullmatch(url)
        assert match, "regex pattern matching failed %s" % url
        
        titles[i] = match.group(1)
        
        country = match.group(2)
        if country:
            countries[i] = country[:-1]
        else:
            countries[i] = 'na'
            
        sites[i] = match.group(3)
        
        agents[i] = match.group(4)
        accesses[i] = match.group(5)
        
    df = pd.DataFrame({
        'page': urls,
        'title': titles,
        'agent': agents,
        'access': accesses,
        'site': sites,
        'country': countries,
    })
    df = df.set_index('page')
    return df

# Why does one want to normalize dummy/one-hot encoded features? For regularized systems, we want the penalization to be fair for all features, thus we generally want to standardize dummies as well.

def page_extracts(extracts, normalize=True):
    '''
    Returns a dictionary with a np array of one-hot encoded features.
    Arrays are np.float64 sparse
    '''
    label = LabelEncoder()
    one_hot = OneHotEncoder()
    def one_hot_encode(col):
        values = extracts[col].values.ravel()
        int_features = label.fit_transform(values).reshape(-1, 1)
        dummies = one_hot.fit_transform(int_features).toarray()
        if normalize:
            dummies = standard_scale(dummies)
        return dummies
    return {str(col): one_hot_encode(col) for col in extracts}


def days_of_week(start_date, end_date):
    prediction_range = pd.date_range(start_date, end_date)
    days_week = prediction_range.dayofweek
    return days_week



def run():
    '''
    Preprocess data into Pytorch Tensors
    '''
    # Load data, get nan positions, get start and end indexes for each tseries
    df, nans, starts, ends = get_clean_data(0.3)

    # calculate our working date range
    start_date, end_date = df.columns[0], df.columns[-1]

    # calculate and make space for prediction window
    forecast_days = 60
    prediction_window = end_date + pd.Timedelta(forecast_days, unit='D')


    # calculate autocorrelation at specified days
    lags = [int(round(365/4)), 365] # yearly and quarterly
    autocorr = get_autocorr(df.values, lags, starts, ends, 1.5, normalize=True)

    # extract page features
    extracts = extract_from_url(df.index.values)
    extracts.drop(['title'], axis=1, inplace=True)
    dic = page_extracts(extracts)

    # get days of week for the prediction window
    dow = days_of_week(start_date, prediction_window)

    page_median = df.median(axis=1)
    standard_scale(page_median)

    page_avg = df.mean(axis=1)
    standard_scale(page_avg)

if __name__=="__main__":
    run()


