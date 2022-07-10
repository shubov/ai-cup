import copy
from pathlib import Path
from time import time
import warnings
from datetime import datetime

# evil parent directory import
import sys
import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import torch

from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from tft import BLTft
from pytorch_forecasting.data import EncoderNormalizer
from pytorch_forecasting.metrics import SMAPE, PoissonLoss, QuantileLoss, MAPE
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
from data.starting_kit.ts_split import GroupedTimeSeriesSplit
from pytorch_forecasting.metrics.base_metrics import MultiHorizonMetric
import matplotlib.pyplot as plt
from data_prep import get_dataframe
from argparse import ArgumentParser

import warnings 
warnings.filterwarnings("ignore")

class QMAPE(MultiHorizonMetric):
    """
    Mean absolute percentage. Assumes ``y >= 0``.

    Defined as ``(y - y_pred).abs() / y.abs()``
    """
    def loss(self, y_pred, target):
        #print(f"pred shape: {y_pred.shape}")
        #print(f"trg shape : {target.shape}")
        loss = (self.to_prediction(y_pred) - target).abs() / (target.abs() + 1e-8)
        return loss
    
    def to_prediction(self, y_pred: torch.Tensor) -> torch.Tensor:
        return y_pred.squeeze(-1)
    
    def to_quantiles(self, y_pred: torch.Tensor) -> torch.Tensor:
        return y_pred

class QRMSE(MultiHorizonMetric):
    """
    Root mean square error

    Defined as ``(y_pred - target)**2``
    """

    def __init__(self, reduction="sqrt-mean", **kwargs):
        super().__init__(reduction=reduction, **kwargs)

    def loss(self, y_pred, target):
        loss = torch.pow(self.to_prediction(y_pred) - target, 2)
        return loss

    def to_prediction(self, y_pred: torch.Tensor) -> torch.Tensor:
        return torch.mean(y_pred, dim=-1)



def eval(device=None, freq='D', mode="worst", versions=(39, 49), ds_path=None):

    if not device:
        DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        DEVICE = device
    
    if freq == 'D':
        m = 1
        data = pd.read_pickle(ds_path or "data/data_daily_melted_sliced_normalized_idd.pkl")
        path = f"lightning_logs/lightning_logs/version_{versions[0]}/checkpoints/"
        checkpoint = os.listdir(path)[0]
        model_path = path + checkpoint
    else:
        data = pd.read_pickle(ds_path or "data/data_hourly_melted_sliced_normalized_filtered_idd.pkl")
        path = f"lightning_logs/lightning_logs/version_{versions[1]}/checkpoints/"
        checkpoint = os.listdir(path)[0]
        model_path = path + checkpoint
        m = 24

    PRED_LENGTH = 7 * m
    KNOWN_LENGTH = (38 - 7) * m
    training = TimeSeriesDataSet(
        data,
        time_idx="time_idx",
        target="consumption",
        group_ids=["ps_id"],
        min_encoder_length=KNOWN_LENGTH // 2,  # keep encoder length long (as it is in the validation set)
        max_encoder_length=KNOWN_LENGTH,
        min_prediction_length=PRED_LENGTH // 2,
        max_prediction_length=PRED_LENGTH,
        static_reals=["n_dwellings"],
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_reals=["consumption"],
        target_normalizer=EncoderNormalizer(),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        #allow_missing_timesteps =True,
    )
    validation = TimeSeriesDataSet.from_dataset(training, data, predict=True, stop_randomization=True)
    batch_size = 128  # set this between 32 to 128
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)

    best_tft = BLTft.load_from_checkpoint(model_path)
    best_tft.to(DEVICE)
    actuals = torch.cat([y[0] for x, y in iter(val_dataloader)]).to(DEVICE)
    predictions = best_tft.predict(val_dataloader).to(DEVICE)
    loss = MAPE(reduction="none")

    error = loss(actuals, predictions).mean(1)
    print(f"predicition error: {error.mean(0)}")
    raw_predictions, x = best_tft.predict(val_dataloader, mode="raw", return_x=True)

    indices = error.argsort(descending=mode=="worst")
    for idx in range(30):  # plot 30 examples
        fig = best_tft.plot_prediction(x, raw_predictions, idx=indices[idx], add_loss_to_title=MAPE(quantiles=best_tft.loss.quantiles))
        plt.show()

def predict(freq='D', device=None, versions=(39, 49), in_f=None, out_f=None):
    
    if not device:
        DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        DEVICE = device
        
    if freq == 'D':
        m = 1
        path = f"lightning_logs/lightning_logs/version_{versions[0]}/checkpoints/"
        checkpoint = os.listdir(path)[0]
        model_path = path + checkpoint
    else:
        path = f"lightning_logs/lightning_logs/version_{versions[1]}/checkpoints/"
        checkpoint = os.listdir(path)[0]
        model_path = path + checkpoint
        m = 24

    PRED_LENGTH = 7 * m
    KNOWN_LENGTH = (38 - 7) * m

    best_tft = BLTft.load_from_checkpoint(model_path)
    best_tft.to(DEVICE)
    counts = pd.read_csv("data/starting_kit/counts.csv")
    og_data = get_dataframe(freq, path=in_f)
    
    tscv = GroupedTimeSeriesSplit(train_window= 38*m, test_window=7*m, train_gap = 0, freq=freq)

    i = 0
    for train_ind, test_ind in tscv.split(og_data, y=og_data, dates = og_data.index):
        if i >= 21:
            print("not sure why its not working after this...")
            break
        print(f"going through samples: #{i}")
        i += 1
        j = 0 
        for index, item in counts.T.iteritems():
            print(f"predicting for dwelling: #{j}")
            id = item["pseudo_id"]
            count = item["n_dwellings"]

            sample = pd.DataFrame({"consumption" : og_data.T.iloc[index][train_ind] / count, "time_idx": train_ind, "n_dwellings": [count for _ in range((len(train_ind)))], "ps_id": [str(id) for _ in range((len(train_ind)))]})
            sample = sample[PRED_LENGTH:]

            r_pred, s_x = best_tft.predict(sample, mode="prediction", return_x=True)
            s_pred = r_pred * count
            s_pred =  s_pred = s_pred.squeeze(0).cpu()
            og_data.T.iloc[index][test_ind] = s_pred.numpy()
            j += 1

    # do the last slice
    l = i * (38*m + 7*m)
    ext_sample_len = KNOWN_LENGTH // 2 + PRED_LENGTH
    #print(ext_sample_len)
    time_ids = [i + l for i in range(-ext_sample_len + 1,1)]
    ext_ids = pd.date_range(start=og_data.index[-1], periods=PRED_LENGTH, freq=freq) + pd.Timedelta(1, unit=freq) 
    extension = {}
    for index, item in counts.T.iteritems():
        id = item["pseudo_id"]
        count = item["n_dwellings"]
        print(f"predicting last slice for dwelling: {index}")
        sample = pd.DataFrame({"consumption" : og_data.T.iloc[index][-ext_sample_len:] / count, "time_idx": time_ids, "n_dwellings": [count for _ in  range(ext_sample_len)], "ps_id": [str(id) for _ in range(ext_sample_len)]})
        r_pred, s_x = best_tft.predict(sample, mode="prediction", return_x=True)
        s_pred = r_pred * count
        s_pred = s_pred.squeeze(0).cpu()
        extension[index] = s_pred.numpy()
    
    og_data = pd.concat([og_data, pd.DataFrame(extension, ext_ids)])

    print("writing predictions...")
    og_data.to_csv(out_f or f"predictions_{'daily' if freq=='D' else 'hourly'}.csv")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-D", action="store_true", help="predict for daily case")
    parser.add_argument("-H", action="store_true", help="predict for hourly case")
    parser.add_argument("-i", type=str, help="path to the original dataset with gaps to predict.")
    parser.add_argument("-o", type=str, help="output path to write prediction results to")
    parser.add_argument("-plot", type=str, help='evalute predictions in plot. options are ["worst", "best"].')
    parser.add_argument("-device", type=str, help="device to use for inference.")
    args = parser.parse_args()
    if args.D and args.H:
        parser.print_help()
        raise Exception("choose either the daily or the hourly case. Not both")
    elif not args.D and not args.H:
        parser.print_help()
        raise Exception("choose at least one case to predict for. options: [-D, -H]")
    else:
        freq = "h" if args.H else 'D'
        if args.plot:
            eval(device=args.device, freq=freq, mode=args.plot, ds_path=args.i)
        else:
            predict(device=args.device, freq=freq, in_f=args.i, out_f=args.o)