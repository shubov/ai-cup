import datetime as dt
import os
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class Dataset:
    """
    Dataset class. Loads consumption and weather data per household group
    Args:
    window_size : No of historical days to include in the window
    """

    def __init__(self,
                 df_path: float,
                 frequency: str = "D",  # change to "H" for second subtask
                 window_size: int = 7,
                 n_rows: int = 100,
                 ids: bool = True,
                 **kwargs) -> None:

        super(Dataset, self).__init__()

        self.path = df_path
        self.nrows = n_rows
        self.window_size = window_size
        self.data = pd.read_csv(os.path.join(os.getcwd(), "", self.path))
        if ids:
            self.data.set_index(self.data.pseudo_id, drop=True, inplace=True)
            self.data.drop(columns='pseudo_id', inplace=True)

        self.data.columns = [dt.datetime.strptime(c, "%Y-%m-%d %H:%M:%S") for c in self.data.columns]
        self.ids = self.data.index
        # self.data = self.data.T[self.data.T.index <= self.weather['min'].index[-1]]
        if frequency == "D":
            self.freq = frequency
            self.data = self.data.T.groupby(self.data.T.index.date).sum().T
        else:
            self.freq = "H"
            self.data = self.data.T.resample(self.freq).sum().T

    def get_test_idx(self) -> object:
        return pd.date_range(start='2017-01-01', end='2019-03-31', freq= self.freq).difference(self.data.T.index)

    def create_lag(self, target, lags=1, thres=0.2):
        """Creates lag features of length window_size"""
        # init scaler
        scaler = StandardScaler()
        df = pd.DataFrame()
        if 0 in lags:
            lags.remove(0)
        for l in lags:
            df[f"lag_{l}"] = target.shift(l)
        # fit scaler
        # features = pd.DataFrame(scaler.fit_transform(df[df.columns]), columns=df.columns)
        features = df
        features.index = target.index
        return features

    def create_ts_features(self, data):

        def get_shift(row):
            """
            3 shifts per day of 8 hours
            """
            if 6 <= row.hour <= 14:
                return 2
            elif 15 <= row.hour <= 22:
                return 3
            else:
                return 1

        data.index = pd.to_datetime(data.index)
        features = pd.DataFrame()
        # features["hour"] = data.index.hour
        features["weekday"] = data.index.weekday
        features["dayofyear"] = data.index.dayofyear
        features["is_weekend"] = data.index.weekday.isin([5, 6]).astype(np.int32)
        # features["weekofyear"] = data.index.isocalendar
        features["month"] = data.index.month
        features["season"] = (data.index.month % 12 + 3) // 3
        features["shift"] = pd.Series(data.index.map(get_shift))
        features["energy use"] = data.values
        features.index = data.index
        return features

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Packs features and target tensors"""
        index = self.ids[idx]
        features = self.data.loc[index]
        lags = self.create_lag(features, lags=range(1, self.window_size + 1), thres=0.2)
        ts = self.create_ts_features(features)
        features_ = ts.join(lags, how="outer").dropna()
        target = features_[features_.index > features.index[self.window_size]]["energy use"]
        features_ = features_[:-1]
        # features_ = features_[features_.index < features_.index[-(self.window_size+1)]]
        # return features_, target
        return torch.tensor(features_.values), torch.tensor(target.values)
