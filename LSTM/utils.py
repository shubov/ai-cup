
from datetime import datetime
from zipfile import ZipFile
import pandas as pd
import numpy as np


def evaluate(y, yhat, perc=True):
    """Method to evaluate MAPE"""

    y = y.drop('pseudo_id', axis = 1).values
    yhat = yhat.drop('pseudo_id', axis = 1).values

    assert y.shape == yhat.shape
    n = len(yhat.index) if type(yhat) == pd.Series else len(yhat)
    for i in range(n):
        error = []
        for a, f in zip(y[i], yhat[i]):
            # avoid division by 0
            if a > 0:
                error.append(np.abs((a - f)/(a)))
        mape = np.mean(np.array(error))
    return mape * 100. if perc else mape


def create_submission(submission_task1, submission_task2, filename: str = None):
    """Method to export the solution"""
    if filename is None:
        filename = 'submission-%s.zip' % str(datetime.now()).replace(' ', '_').replace(':', '-')
    with ZipFile(filename, 'w') as zip:
        zip.writestr("submission_daily.csv", submission_task1.to_csv())
        zip.writestr("submission_hourly.csv", submission_task2.to_csv())

    print('wrote', filename)
