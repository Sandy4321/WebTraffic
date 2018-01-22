from torch.utils.data import Dataset
import os
import pandas as pd
import numpy as np


class WebTrafficDataset(Dataset):
    """WebTraffic dataset."""

    def __init__(self, root_dir, file_base, transform=None):
        """
        Args:
            file_base (string): Basename of the file with annotations.
            root_dir (string): Directory with all the traffic data.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform

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
        df.columns = df.columns.astpye('M8[D]')

        self.dataframe = df

    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, idx):
        sample = self.dataframe.iloc[idx]

        return sample

#########################
# TRANSFORMS
#########################


df = WebTrafficDataset('data', 'train_1')
entry0 = df[0]
