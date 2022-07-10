from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.optim import Ranger
import torch
import numpy as np
from torch.optim.lr_scheduler import _LRScheduler, LambdaLR, ReduceLROnPlateau


class WarmupScheduler(_LRScheduler):
    def __init__(self, 
                 optimizer: torch.optim.Optimizer,
                 dim_embed: int,
                 warmup_steps: int,
                 last_epoch: int=-1,
                 verbose: bool=False) -> None:

        self.dim_embed = dim_embed
        self.warmup_steps = warmup_steps
        self.num_param_groups = len(optimizer.param_groups)

        super().__init__(optimizer, last_epoch, verbose)
        
    def get_lr(self) -> float:
        lr = calc_lr(self._step_count, self.dim_embed, self.warmup_steps)
        return [lr] * self.num_param_groups


def calc_lr(step, dim_embed, warmup_steps):
    return dim_embed**(-0.5) * min(step**(-0.5), step * warmup_steps**(-1.5))
    
# not really a lot of modifications
# add Warmup Scheduler option to TemporalFusion Transformer.

class BLTft(TemporalFusionTransformer):

    def __init__(self, warmup_steps=None, **kwargs):
        super().__init__(**kwargs)
        self.hparams.update({'warmup_steps': warmup_steps})

    def configure_optimizers(self):
            """
            Configure optimizers.

            Uses single Ranger optimizer. Depending if learning rate is a list or a single float, implement dynamic
            learning rate scheduler or deterministic version

            Returns:
                Tuple[List]: first entry is list of optimizers and second is list of schedulers
            """

            print(f"self.hparams.warmup_steps: {self.hparams.warmup_steps}")
            # either set a schedule of lrs or find it dynamically
            if self.hparams.optimizer_params is None:
                optimizer_params = {}
            else:
                optimizer_params = self.hparams.optimizer_params
            # set optimizer
            lrs = self.hparams.learning_rate
            if isinstance(lrs, (list, tuple)):
                lr = lrs[0]
            else:
                lr = lrs
            if callable(self.optimizer):
                try:
                    optimizer = self.optimizer(
                        self.parameters(), lr=lr, weight_decay=self.hparams.weight_decay, **optimizer_params
                    )
                except TypeError:  # in case there is no weight decay
                    optimizer = self.optimizer(self.parameters(), lr=lr, **optimizer_params)
            elif self.hparams.optimizer == "adam":
                optimizer = torch.optim.Adam(
                    self.parameters(), lr=lr, weight_decay=self.hparams.weight_decay, **optimizer_params
                )
            elif self.hparams.optimizer == "adamw":
                optimizer = torch.optim.AdamW(
                    self.parameters(), lr=lr, weight_decay=self.hparams.weight_decay, **optimizer_params
                )
            elif self.hparams.optimizer == "ranger":
                optimizer = Ranger(self.parameters(), lr=lr, weight_decay=self.hparams.weight_decay, **optimizer_params)
            elif self.hparams.optimizer == "sgd":
                optimizer = torch.optim.SGD(
                    self.parameters(), lr=lr, weight_decay=self.hparams.weight_decay, **optimizer_params
                )
            elif hasattr(torch.optim, self.hparams.optimizer):
                try:
                    optimizer = getattr(torch.optim, self.hparams.optimizer)(
                        self.parameters(), lr=lr, weight_decay=self.hparams.weight_decay, **optimizer_params
                    )
                except TypeError:  # in case there is no weight decay
                    optimizer = getattr(torch.optim, self.hparams.optimizer)(self.parameters(), lr=lr, **optimizer_params)
            else:
                raise ValueError(f"Optimizer of self.hparams.optimizer={self.hparams.optimizer} unknown")

            # set scheduler
            if isinstance(lrs, (list, tuple)):  # change for each epoch
                # normalize lrs
                lrs = np.array(lrs) / lrs[0]
                scheduler_config = {
                    "scheduler": LambdaLR(optimizer, lambda epoch: lrs[min(epoch, len(lrs) - 1)]),
                    "interval": "epoch",
                    "frequency": 1,
                    "strict": False,
                }
            if self.hparams.reduce_on_plateau_patience is None:
                scheduler_config = {}
            
            if self.hparams.warmup_steps is not None:
                print("using warmup scheduler")
                scheduler_config = {
                    "scheduler": WarmupScheduler(
                        optimizer,
                        dim_embed=self.hparams.hidden_size,
                        warmup_steps=self.hparams.warmup_steps,
                    ),
                    "monitor": "val_loss",  # Default: val_loss
                    "interval": "epoch",
                    "frequency": 1,
                    "strict": False,
                }
            else:  # find schedule based on validation loss
                scheduler_config = {
                    "scheduler": ReduceLROnPlateau(
                        optimizer,
                        mode="min",
                        factor=1.0 / self.hparams.reduce_on_plateau_reduction,
                        patience=self.hparams.reduce_on_plateau_patience,
                        cooldown=self.hparams.reduce_on_plateau_patience,
                        min_lr=self.hparams.reduce_on_plateau_min_lr,
                    ),
                    "monitor": "val_loss",  # Default: val_loss
                    "interval": "epoch",
                    "frequency": 1,
                    "strict": False,
                }
            return {"optimizer": optimizer, "lr_scheduler": scheduler_config}