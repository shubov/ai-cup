# BLTFT
The Blöde Liga Temporal Fusion Transformer (which is actually just a regular temporal fusion transformer with a warmup LR Scheduler).

## Installation
Point your working directory to ./transformer
```bash
cd ./transformer
```

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install requirements.txt.

```bash
pip install -r requirements.txt
```

# Usage

Step 1 is to prepare data.

## Creating necessary features
You can use data_prep.py from the commandline to run a variety of operations on the original data.

```bash
python data_prep.py [options]
```

To create a pickled dataframe that has the format and additional features needed to train the Temporal Fusion Transformer simply run:

```bash
python data_prep.py -melt -D -i "/path/to/data.csv" -o "/path/to/processed_data.pkl"
```

This will write the pickled data for a DAILY (-D) view of the data in data.csv to processed_data.pkl.
For the hourly data replace -D with -H like this:

```bash
python data_prep.py -melt -H -i "/path/to/data.csv" -o "/path/to/processed_data.pkl"
```

## Filtering
To additionally filter the daily or hourly data before pickeling add -f:

```bash
python data_prep.py -melt -H -i "/path/to/data.csv" -o "/path/to/processed_data.pkl" -f
```

Filtering applies two different operations on the input data.
1: Clip very low readings (anything in the bottom 1% of the data is removed and replaced by a linear interpolation between the value before and after the interval that is cut out).
2: Negative gradient filtering. Roughly: If the relative difference between a datapoint and its predecessor is too large it is dropped and replaced by linear interpolation. This happens only for downward spikes, because we assume a faulty or missed reading.

the filter functions are applied as follows in sequence.
-> clipping -> gradient filter -> clipping again (because the data will be renormalized).

Filtering has safety mechanisms built in so if there is a lot of data removed a plot will show up to inspect what data caused issues with the filtering mechanisms.
We assume there is something wrong with the filtering mechanism in this case and disregard the filter output using the original data instead.

## Train

temporal_fusion.py holds utilies for training, finetuning/training continuation, hyperparameter tuning and learning rate estimation.
From this point forward we will only show the process for the hourly case. The daily case can be run equivalently by replacing -H with -D
Also the scripts used beyond this point allow to specify a pytorch lightning accelerator a.k.a. device.
For example running training on GPU1 of a system yields the argument -device "cuda:1". If this argument is not provided the FIRST cuda capable GPU in the system will be automatically chosen if torch recognizes it, otherwise the CPU.
We will also omit this in the following.

### LR Estimation
This can be skipped if the default model parameters in temporal_fusion.py have not been changed.
First estimate the best learning rate for the model and data.
Depending on the accelerator hardware available in your system this function may or may not be supported.

```bash
python temporal_fusion.py -lr -H
```

The result of lr estimation will be shown as a plot.
If you choose to use it replace the value of learning_rate in the train method of temporal_fusion.py. This is not required. The script comes with good learning rate estimations set as default.

### Training

To train the model make sure you have the correct dataset for -D or -H prepared somewhere.

```bash
python temporal_fusion.py -t -H -ds-path "/path/to/dataset.pkl" -epochs 5000 
```

logs and checkpoints will be output to ./lightning_logs
If tensorboard is installed you can run:

```bash
tensorboard --logdir=lightning_logs
```

to see the training progress.

## Evaluation and Prediction

To evaluate or run inference on a trained checkpoint of a model you can use eval_temporal_fusion.py

### Evaluate model accuracy
Run eval_temporal_fusion in eval mode.
In this case we use the path to the previously created pickled dataset to get a validation dataset

```bash
python eval_temporal_fusion.py -H -i "train_filtered_melted.pkl" -plot "worst"
```

To plot out the 30 worst predictions from the validation set. Or to get the 30 best:

```bash
python eval_temporal_fusion.py -H -i "train_filtered_melted.pkl" -plot "best"
```

### Inference
Depending on what data we trained on (filtered or unfiltered) we point the script to the correct dataset.csv that still contains the missing time slices.

```bash
python eval_temporal_fusion.py -H -i "train_filtered.csv"
```

This runs through the original/filtered dataset and fill in the gaps (train indices) as provided by GroupedTimeSeriesSplit
Writes the output to predictions_hourly.csv / predictions_daily.csv respectively.

## Creating submissions

The data_prep.py utility provides a function to package predictions into the submission format.
Run:

```bash
python data_prep.py -H -s -i "path/to/predictions.csv" -o "submission.csv"
```