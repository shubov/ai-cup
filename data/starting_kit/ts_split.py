from typing import Union
import numpy as np
import pandas as pd
from pandas.core.dtypes.common import is_datetime_or_timedelta_dtype
from sklearn.model_selection._split import _BaseKFold


class GroupedTimeSeriesSplit(_BaseKFold):
    """Time Series cross-validator

    Provides train/test indices to split time series data samples
    that are observed at fixed time intervals, in train/test sets.
    In each split, test indices must be higher than before, and thus shuffling
    in cross validator is inappropriate. Test splits are treated preferentially,
    i.e. if there are not enough samples the test split is filled first.
    This splitter works on a date range, which is not necessarily aligned to the
    available data if it has missing days.

    ----------

    train_window : int, default= 38
        Maximum size for a single training set.

    test_window : int, default= 7
        Used to set the size of the test set.

    train_gap : int, default= 0
        Gap (in days) before the training set.

    test_gap : int, default=0
        Gap (in days) between the training and test set.
    """

    def __init__(self, train_window: int = 38, test_window= 7, train_gap: int = 0, test_gap: int = 0):
        self.train_window = train_window
        self.test_window = test_window
        self.train_gap = train_gap
        self.test_gap = test_gap
        self.n_folds_ = None

    def split(self, X: pd.DataFrame, y, dates:Union[pd.Series, np.ndarray], *_):
        """Generate indices to split data into training and test set according to provided dates.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.

        dates : array-like of shape (n_samples,)
            Dates of the samples, can be passed to sklearn via the `groups` parameter.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        """
        if not is_datetime_or_timedelta_dtype(dates):
            dates = pd.to_datetime(dates)

        start_date, end_date = dates.min(), dates.max()
        date_range = pd.date_range(start_date, end_date)
        n_dates = len(date_range)

        indices = np.arange(n_dates)
        train_starts = range(0, n_dates, self.train_window + self.test_window + self.train_gap + self.test_gap)

        self.n_folds_ = len(train_starts)

        for train_start in train_starts:
            avail_days = min(n_dates - train_start,
                             self.train_window + self.test_window + self.train_gap + self.test_gap)
            test_start = max(train_start, train_start + avail_days - self.test_window - self.train_gap)
            train_dates = date_range[indices[train_start: test_start - self.test_gap]]
            test_dates = date_range[test_start: train_start + avail_days - self.train_gap]
            train_indices = np.where(np.isin(dates, train_dates))[0]
            test_indices = np.where(np.isin(dates, test_dates))[0]
            if len(test_dates) < self.test_window:
                continue
            yield list(train_indices), list(test_indices)
