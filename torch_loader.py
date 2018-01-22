import argparse
import os
import pandas as pd
import numpy as np

from torch.utils.data import Dataset

import preprocessing as pre


class WebTrafficDataset(Dataset):
    """WebTraffic dataset."""

    def __init__(self, root_dir, file_base, threshold=1.0,
            forecast_days=60, start=None, end=None):
        """
        Args:
            file_base (string): Basename of the file with annotations.
            root_dir (string): Directory with all the traffic data.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir

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

        # calculate our working date range
        start_date, end_date = self.df.columns[0], self.df.columns[-1]

        # calculate and make space for prediction window
        prediction_window = end_date + pd.Timedelta(forecast_days, unit='D')

        # calculate autocorrelation at specified days
        year_corr = pre.batch_autocorrelation(self.df, 365, starts, ends, 1.5)
        pre.undefined_corr_pct(year_corr, 365)
        quarter_corr = pre.batch_autocorrelation(self.df, 91, starts, ends, 2.0)
        pre.undefined_corr_pct(quarter_corr, 91)

        # extract page features
        extracts = pre.extract_from_url(self.df.index.values)
        extracts.drop(['title'], axis=1, inplace=True)
        # one hot encoded features
        regex_features = pre.page_extracts(extracts)

        # get days of week for the prediction window
        dow = pre.days_of_week(start_date, prediction_window)

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

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        sample = self.df.iloc[idx]
        agent = self.agents[idx]
        country = self.countries[idx]
        site = self.site[idx]
        median = self.page_median[idx]
        quarterly = self.quarterly[idx]
        yearly = self.yearly[idx]
        dow = self.days_of_week[idx]
        return sample, agent, country, site, median, quarterly, yearly, dow


if __name__ == "__main__":
    # TODO parse arguments to initialize DATASET class   
    parser = argparse.ArgumentParser(description='Prepare data')
    parser.add_argument('data_dir')
    parser.add_argument('file_base')
    parser.add_argument('--threshold', default=0.0, type=float, help="Series minimal length threshold (pct of data length)")
    parser.add_argument('--forecast_days', default=60, type=int, help="Add N days in a future for prediction")
    parser.add_argument('--start', help="Effective start date. Data before the start is dropped")
    parser.add_argument('--end', help="Effective end date. Data past the end is dropped")
    args = parser.parse_args()

    ds = WebTrafficDataset(args.data_dir, args.file_base, args.forecast_days,
            args.start, args.end)
    # TODO pickle dataset
    # TODO dataloader
    # TODO to tensor transformation
