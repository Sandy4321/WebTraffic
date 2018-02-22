import argparse
import os
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.autograd import Variable

import preprocessing as pre


class WebTrafficDataset(Dataset):
    """WebTraffic dataset."""

    def __init__(self, root_dir, file_base, threshold=0.0,
            forecast_days=60, lags=30, start=None, end=None, transform=None):
        """
        Args:
            file_base (string): Basename of the file with annotations.
            root_dir (string): Directory with all the traffic data.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.file_base = file_base
        self.lags = lags

        # load the dataset
        PKL_PATH = os.path.join(root_dir, file_base + '.pkl')
        if os.path.exists(PKL_PATH):
            print('Loading pickle...')
            df = pd.read_pickle(PKL_PATH)
            print('Done!')
        else:
            CSV_PATH = os.path.join(root_dir, file_base + '.csv')
            print('Loading csv...')
            df = pd.read_csv(CSV_PATH)
            print('Exporting to pickle')
            df.to_pickle(PKL_PATH)
            print('Done!')
        df.set_index('Page', inplace=True)
        df.sort_index(inplace=True)
        df.columns = df.columns.astype('M8[D]')

        self.df = df

        # pre-process data
        # TODO - is this an elegant implementation? I cant see a way of
        # adding features with transformations

        # normalize dataset, get nans, starts and ends
        df, nans, starts, ends = pre.filtered_log1p(self.df,
                threshold=threshold, start=start, end=end)	
        self.df = df

        # calculate our working date range
        start_date, end_date = self.df.columns[0], self.df.columns[-1]

        # calculate and make space for prediction window
        prediction_window = end_date + pd.Timedelta(forecast_days, unit='D')

        # calculate autocorrelation at specified days
        year_corr = pre.batch_autocorrelation(self.df.values, 365, starts, ends, 1.5)
        pre.undefined_corr_pct(year_corr, 365)
        quarter_corr = pre.batch_autocorrelation(self.df.values, 91, starts, ends, 2.0)
        pre.undefined_corr_pct(quarter_corr, 91)

        # extract page features
        extracts = pre.extract_from_url(self.df.index.values)
        extracts.drop(['title'], axis=1, inplace=True)
        # one hot encoded features
        regex_features = pre.page_extracts(extracts)

        # get days of week for the prediction window
        dow = pre.days_of_week(start_date, prediction_window)
        dow = pre.standard_scale(dow)

        page_median = df.median(axis=1)
        page_median = pre.standard_scale(page_median)

        # reset nans and save meaningful attributes
        self.df[nans] = np.nan
        self.agents = regex_features['agent']
        self.countries = regex_features['country']
        self.access = regex_features['access']
        self.site = regex_features['site']
        self.page_median = page_median
        self.yearly = year_corr
        self.quarterly = quarter_corr
        self.day_of_week = dow

        self.start_date = start_date
        self.end_date = end_date
        self.prediction_window = prediction_window
        self.starts = starts
        self.ends = ends

        print('Done loading and preprocessing set!')

    def batch_series(self, data:np.ndarray, lags=1, drop_nan=True):
        cols = []
        for i in range(len(data)):
            tmp = pre.shift(data, lags-i)
            # extract number of lags
            cols += [tmp[:lags]]
        series = np.array(cols)

        #print(data.shape)
        target = np.array([data[i][0] for i in range(data.shape[0])])

        # drop batches that have a nan value in the series position
        if drop_nan:
            size = series.shape[0]
            # build mask for first sliding window nan values
            mask = np.full(size, True, dtype='bool')
            for i in range(size):
                if np.isnan(series[i][0][0]):
                    mask[i] = False
                else:
                    break
            series = series[mask]
            target = target[mask]

        return series, target

    def get_series(self, idx):
        return self.df.iloc[idx].values

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        tseries = self.get_series(idx)
        agent = self.agents[idx]
        access = self.access[idx]
        country = self.countries[idx]
        site = self.site[idx]
        median = self.page_median.iloc[idx]
        quarterly = self.quarterly[idx]
        yearly = self.yearly[idx]
        dow = self.day_of_week

        # Assemble features into a stacked vector [len(tseries) x #features]
        x_dow = dow[:len(tseries)]
        y_dow = dow[len(tseries):]
        
        stack = np.stack((yearly, quarterly, median))
        concat = np.concatenate((agent, site, country, stack))
        
        result = np.tile(concat, [len(tseries), 1]).transpose()
        series = np.vstack((tseries, x_dow, result)).transpose()
        #series = np.array(np.array(tseries)).transpose()

        # Batch stacked vector into the number of required lags
        datapoint, target = self.batch_series(series, lags=self.lags)
        #datapoint = Variable(torch.from_numpy(datapoint))
        #target = Variable(torch.from_numpy(target))

        # we need to concat every feature into a single tensor
        #datapoint = {
        #        'tseries': tseries,
        #        'agent': agent,
        #        'country': country,
        #        'site': site,
        #        'access': access,
        #        'median': median,
        #        'quarterly': quarterly,
        #        'yearly': yearly,
        #        'dow': dow,
        #        }
        return datapoint, target



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare data')
    parser.add_argument('data_dir')
    parser.add_argument('file_base')
    parser.add_argument('--threshold', default=0.0, type=float, help="Series minimal length threshold (pct of data length)")
    parser.add_argument('--forecast_days', default=60, type=int, help="Add N days in a future for prediction")
    parser.add_argument('--lags', default=30, type=int, help="N days for the sliding window.")
    parser.add_argument('--start', help="Effective start date. Data before the start is dropped")
    parser.add_argument('--end', help="Effective end date. Data past the end is dropped")
    args = parser.parse_args()

    ds = WebTrafficDataset(args.data_dir, args.file_base, args.threshold, args.forecast_days, args.lags,
        args.start, args.end)
    torch.save(ds, 'data/dataset.pt')
