import copy
from pathlib import Path
import warnings
from datetime import datetime
import os

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from pytorch_forecasting import Baseline, EncoderNormalizer, TimeSeriesDataSet #, TemporalFusionTransformer
from tft import BLTft
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import SMAPE, PoissonLoss, QuantileLoss, MAPE
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
from pytorch_forecasting.metrics.base_metrics import MultiHorizonMetric
from data_prep import get_dataframe, prepare_features
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
        return torch.mean(y_pred, dim=-1)#y_pred.squeeze(-1)
    
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

def get_device_and_accelerator(device=None):
    if not device:
        ACC = "gpu" if torch.cuda.is_available() else "cpu"
        DEVICE = ["cuda:0"] if torch.cuda.is_available() else os.cpu_count()
    else:
        if device == "cpu":
            ACC = "cpu"
            DEVICE = os.cpu_count()
        else:
            ACC = "gpu"
            DEVICE = [device]
    return DEVICE, ACC

def test_dataloader(freq='D', ds_path=None):
    if freq == 'D':
        m = 1
        data = pd.read_pickle(ds_path or "data/data_daily_melted_sliced_normalized_idd.pkl")
    else:
        data = pd.read_pickle(ds_path or "data/data_hourly_melted_sliced_normalized_filtered_idd.pkl")
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
    sample = next(iter(train_dataloader))#
    src = sample[0]["encoder_target"]
    tgt = sample[1][0]
    print(src)
    print(tgt)
    print(src.shape)
    print(tgt.shape)

def train(lr_evaluation=False, freq='D', epochs=4000, device=None, ds_path=None):
    if freq == 'D':
        m = 1
        data = pd.read_pickle(ds_path or "data/data_daily_melted_sliced_normalized_idd.pkl")
        gradient_clip_val = 0.1
        hidden_size = 128
        dropout = 0.1 
        hidden_continuous_size = 32
        attention_head_size = 1
        learning_rate = 0.04
        reduce_on_plateau_patience = None
        warmup_steps = 20 # warm up for 20 epochs
        output_size = 32
    else:
        data = pd.read_pickle(ds_path or "data/data_hourly_melted_sliced_normalized_filtered_idd.pkl")
        m = 24
        gradient_clip_val = 0.06308174838920966
        hidden_size = 256
        dropout = 0.15 
        hidden_continuous_size = 64
        attention_head_size = 2
        learning_rate = 0.0038 #0.0015127821568719202
        reduce_on_plateau_patience = None
        warmup_steps = 20 # warm up for 20 epochs
        output_size = 64

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

    DEVICE, ACC = get_device_and_accelerator(device)


    #qs = [0.005, 0.01, 0.05, 0.1, 0.125, 0.15, 0.175, 0.2, 0.25, 0.3, 0.3235294117647059, 0.35294117647058826, 0.38235294117647056, 0.4117647058823529, 0.4411764705882353, 0.47058823529411764, 0.5, 0.5294117647058824, 0.5588235294117647, 0.5882352941176471, 0.6176470588235294, 0.6470588235294118, 0.6764705882352942, 0.7, 1.0 - 0.25, 1.0 - 0.2, 1.0 - 0.175, 1.0 - 0.15, 1.0 - 0.125, 1.0 - 0.1, 1 - 0.05, 1 -0.01, 1.0 - 0.005]
    loss_fn = QuantileLoss(quantiles=[0.001, 0.01, 0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98, 0.99, 0.999])
    validation = TimeSeriesDataSet.from_dataset(training, data, predict=True, stop_randomization=True)
    batch_size = 128  # set this between 32 to 128
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)

    actuals = torch.cat([y for x, (y, weight) in iter(val_dataloader)])
    baseline_predictions = Baseline().predict(val_dataloader)
    baseline = (actuals - baseline_predictions).abs().mean().item()
    print(f"baseline predicition error: {baseline}")

    ############ Find learning rate ##############
    pl.seed_everything(42)
    if lr_evaluation:
        trainer = pl.Trainer(
            accelerator= ACC,
            # clipping gradients is a hyperparameter and important to prevent divergance
            # of the gradient for recurrent neural networks
            gradient_clip_val=gradient_clip_val,
            devices=DEVICE,
        )

        tft = BLTft.from_dataset(
            training,
            # not meaningful for finding the learning rate but otherwise very important
            learning_rate=learning_rate,
            hidden_size=hidden_size,  # most important hyperparameter apart from learning rate
            # number of attention heads. Set to up to 4 for large datasets
            attention_head_size=attention_head_size,
            #gradient_clip_val = gradient_clip_val,
            dropout=dropout,  # between 0.1 and 0.3 are good values
            hidden_continuous_size=hidden_continuous_size,  # set to <= hidden_size
            output_size=output_size,  # 7 quantiles by default
            loss=loss_fn,
            warmup_steps=warmup_steps,
            # reduce learning rate if no improvement in validation loss after x epochs
            reduce_on_plateau_patience=reduce_on_plateau_patience,
        )
        print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")


        res = trainer.tuner.lr_find(
            tft,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
            max_lr=10.0,
            min_lr=1e-6,
        )

        print(f"suggested learning rate: {res.suggestion()}")
        fig = res.plot(show=True, suggest=True)
        fig.show()
    else:
        ############# TRAINING #################

        # configure network and trainer
        #early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-6, patience=150, verbose=False, mode="min") # mayb don't use this with warmup scheduler
        lr_logger = LearningRateMonitor()  # log the learning rate
        logger = TensorBoardLogger("lightning_logs")  # logging results to a tensorboard

        trainer = pl.Trainer(
            max_epochs=epochs,
            weights_summary="top",
            gradient_clip_val=gradient_clip_val,
            limit_train_batches=40,  # coment in for training, running valiation every 30 batches
            # fast_dev_run=True,  # comment in to check that networkor dataset has no serious bugs
            callbacks=[lr_logger], #early_stop_callback],
            logger=logger,
            accelerator=ACC,
            devices=DEVICE,
        )


        tft = BLTft.from_dataset(
            training,
            learning_rate=learning_rate,
            hidden_size=hidden_size,
            attention_head_size=attention_head_size,
            dropout=dropout,
            hidden_continuous_size=hidden_continuous_size,
            output_size=output_size,  # 7 quantiles by default
            loss=loss_fn,
            log_gradient_flow=False,
            log_interval=10,  # uncomment for learning rate finder and otherwise, e.g. to 10 for logging every 10 batches
            reduce_on_plateau_patience=reduce_on_plateau_patience,
            warmup_steps = warmup_steps
        )
        print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")


        trainer.fit(
            tft,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )

def finetune(version, device=None, freq='D', epochs=4000, ds_path=None):
    path = f"lightning_logs/lightning_logs/version_{version}/checkpoints/"
    checkpoint = os.listdir(path)[0]
    checkpoint_path = path + checkpoint

    if freq == 'D':
        m = 1
        data = pd.read_pickle(ds_path or "data/data_daily_melted_sliced_normalized_idd.pkl")
    else:
        data = pd.read_pickle(ds_path or "data/data_hourly_melted_sliced_normalized_filtered_idd.pkl")
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

    gradient_clip_val = 0.06308174838920966

    DEVICE, ACC = get_device_and_accelerator(device)

    batch_size=128
    lr_logger = LearningRateMonitor()  # log the learning rate
    logger = TensorBoardLogger("lightning_logs")  # logging results to a tensorboard
    validation = TimeSeriesDataSet.from_dataset(training, data, predict=True, stop_randomization=True)
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)


    trainer = pl.Trainer(
        max_epochs=epochs,
        weights_summary="top",
        gradient_clip_val=gradient_clip_val,
        limit_train_batches=40,  # coment in for training, running valiation every 30 batches
        # fast_dev_run=True,  # comment in to check that networkor dataset has no serious bugs
        callbacks=[lr_logger],
        logger=logger,
        accelerator=ACC,
        devices=DEVICE,
    )

    tft = BLTft.load_from_checkpoint(checkpoint_path)
    optimizer = torch.optim.RAdam(tft.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=tft.hparams.weight_decay)
    # use warmup scheduler to kick network out of its current local minimum a little bit
    scheduler_config = {
                    "scheduler": CosineAnnealingWarmRestarts(optimizer, 100, 1.2),
                    "interval": "epoch",
                    "frequency": 1,
                    "strict": False,
                    } 
    trainer.init_optimizers(optimizer, scheduler_config)
    trainer.fit(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    

def hyperparam_search(freq='D', ds_path=None):

    import pickle

    from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
    if freq == 'D':
        m = 1
        data = pd.read_pickle(ds_path or "data/data_daily_melted_sliced_normalized_idd.pkl")
    else:
        data = pd.read_pickle(ds_path or "data/data_hourly_melted_sliced_normalized_filtered_idd.pkl")
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
    t_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    v_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)

    # create study
    study = optimize_hyperparameters(
        t_dataloader,
        v_dataloader,
        model_path="optuna_test",
        n_trials=200,
        max_epochs=50,
        gradient_clip_val_range=(0.01, 1.0),
        hidden_size_range=(8, 512),
        hidden_continuous_size_range=(8, 512),
        attention_head_size_range=(1, 8),
        learning_rate_range=(0.001, 0.1),
        dropout_range=(0.1, 0.3),
        trainer_kwargs=dict(limit_train_batches=30),
        reduce_on_plateau_patience=10,
        use_learning_rate_finder=False,  # use Optuna to find ideal learning rate or use in-built learning rate finder
    )

    # save study results - also we can resume tuning at a later point in time
    with open("test_study.pkl", "wb") as fout:
        pickle.dump(study, fout)

    # show best hyperparameters
    print(study.best_trial.params)

if __name__ == "__main__":
    #hyperparam_search()
    parser = ArgumentParser()
    parser.add_argument("-t", action="store_true", help="run training. Requires to specify -h or -d (frequency parameter).")
    parser.add_argument("-s", action="store_true", help="run hyperparameter search. Requires to specify -h or -d (frequency parameter).")
    parser.add_argument("-device", type=str, help="device. cuda:0, cuda:1, cpu etc. Default is cpu (None)")
    parser.add_argument("-f", action="store_true", help="run finetuning on checkpoint. Requires to specify version number of checkpoint in lightning log directory.")
    parser.add_argument("-v", type=int, help="Version to finetune. Only valid if option is finetune.")
    parser.add_argument("-lr", action="store_true", help="do a find learning rate run for the given configuration.")
    parser.add_argument("-epochs", type=int, help="how many epochs to run training or finetuning for.")
    parser.add_argument("-H", action="store_true", help="run chosen operation hourly dataset")
    parser.add_argument("-D", action="store_true", help="run chosen operation daily dataset")
    parser.add_argument("-ds-path", type=str, help="path to the dataset directory. The correct dataset for -D and -H will be chosen based on name.")
    args = parser.parse_args()
    if args.D and args.H:
        parser.print_help()
        raise Exception("choose either the daily or the hourly case. Not both")
    elif not args.D and not args.H:
        parser.print_help()
        raise Exception("choose at least one case to predict for. options: [-D, -H]")
    else:
        freq = "h" if args.H else 'D'
        if args.t:
            train(lr_evaluation=args.lr, device=args.device, epochs=args.epochs or 4000, freq=freq, ds_path=args.ds_path)
        if args.s:
            hyperparam_search(freq)
        if args.f:
            finetune(args.v, device=args.device, epochs=args.epochs or 1000, freq=freq, ds_path=args.ds_path)
