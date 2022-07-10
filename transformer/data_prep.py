import pandas as pd
from datetime import datetime

# evil parent directory import
import sys
import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from data.starting_kit.ts_split import GroupedTimeSeriesSplit
from torch.utils.data import Dataset
import torch
from argparse import ArgumentParser
from anomaly_detection import clean_data_lerp
#from main import KNOWN_LENGTH, PRED_LENGTH

def get_dataframe(resolution='D', filtered=True, in_f=None, path=None):
    if path:
        df = pd.read_csv(path)
    else:
        if not in_f:    
            df = pd.read_csv('train' + ('_filtered' if filtered else '') + '.csv')
        else:
            df = pd.read_csv(in_f)
    # drop index for feature preparation
    df_ = df.drop(columns='pseudo_id')
    # convert dates to pandas datetime
    df_.columns = [datetime.strptime(c, "%Y-%m-%d %H:%M:%S") for c in df_.columns]
    # Aggregate energy use values per day
    df_ = df_.T
    df_.index = pd.to_datetime(df_.index)
    
    df_ = df_.resample(resolution).sum()
    
    df_.index.freq = resolution
    df_.index.name = 'date'
    return df_

def prepare_features(df, normalize=False):
    m = 1
    if df.index.freq == 'h':
        m = 24
    tscv = GroupedTimeSeriesSplit(train_window= 38*m, test_window=7*m, train_gap = 0, freq=df.index.freq)
    rm_indices = [] # indices to remove (empty columns)
    sample_indices = []
    i = 0
    for train_ind, test_ind in tscv.split(df, y=df, dates = df.index):
        rm_indices += test_ind
        sample_indices.append((df.index[train_ind], str(i)))
        i += 1
    
    
    df.drop(df.index[rm_indices], inplace=True)
    counts = pd.read_csv('../data/starting_kit/counts.csv')
    mapping_counts = {}
    mapping_ids = {}
    for col in df.columns: # these are the households with their timeseries
        mapping_counts[col] = counts["n_dwellings"][col]
        mapping_ids[col] = str(counts["pseudo_id"][col])
    
    keys = df.columns
    

    df = df.reset_index()
    df = pd.melt(df, id_vars='date', value_vars=keys, var_name='id', value_name='consumption')
    df = df.set_index(pd.DatetimeIndex(df['date']))
    df['n_dwellings'] = df.loc[:,'id']
    df = df.replace({'id': mapping_ids, 'n_dwellings': mapping_counts})
    df = df.rename(columns={'date': 'time_idx'})
    if normalize:
        df["consumption"] /= df["n_dwellings"]
    df['time_idx'] = df['time_idx'].apply(lambda x: x.value)
    df['time_idx'] -= df['time_idx'].min()
    dt = df['time_idx'][1]
    df['time_idx'] = df['time_idx'] // dt
    df['ps_id'] = df.loc[:,'id']
    df = df.set_index(['id'], append=True)
    for time, sample_num in sample_indices:
        df.loc[(time, slice(None)), "ps_id"] += "_" + sample_num
    return df


def pickle_prepared(df, out_f="data.pkl"):
    out_f = out_f[:-4] + ".pkl" # just to be sure
    df.to_pickle(out_f)
    print(f'wrote: {out_f}')

class RBFDataset(Dataset):
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __len__(self):
        return self.x.size(0)
    
    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        return (x, y)

def as_dataset(df, freq):
    m = 1
    if df.index.freq == 'h':
        m = 24
    
    KNOWN_LENGTH = 38*m
    PRED_LENGTH = 7*m
    tscv = GroupedTimeSeriesSplit(train_window=KNOWN_LENGTH, test_window=PRED_LENGTH, train_gap = 0, freq=df.index.freq)
    xs = []
    ys = []
    for train_ind, infer_ind in tscv.split(df, y=df, dates = df.index):
        xs.append(torch.tensor(df.iloc[train_ind[:-PRED_LENGTH]].values, dtype=torch.float32))
        ys.append(torch.tensor(df.iloc[train_ind[-PRED_LENGTH:]].values, dtype=torch.float32))
    
    del xs[-1]
    del ys[-1]
    data_x = torch.stack(xs, dim=0)
    data_y = torch.stack(ys, dim=0)

    return RBFDataset(data_x, data_y)


def make_submission(df, freq, out_f="submission.csv"):
    rf_sub = pd.read_csv("../data/starting_kit/sample_submission_" + ("hourly" if freq == 'h' else "daily") + ".csv")
    new_daterange = pd.date_range(start="2019-04-29", end="2019-09-04 23:00:00", freq=freq)
    rf_sub = rf_sub.drop(columns="pseudo_id")
    
    rf_sub.columns = pd.to_datetime(rf_sub.columns)
    rf_sub = rf_sub.T
    
    rf_sub = rf_sub.reindex(new_daterange)
    rf_sub = rf_sub.dropna(thresh=1)

    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    df = df.reindex(rf_sub.index)
    df = df.T
    df = df.rename_axis(None, axis=1)

    ids = pd.read_csv("../data/starting_kit/counts.csv")
    df = df.set_index(ids["pseudo_id"])

    dt = lambda t: datetime.strptime(t, "%Y-%m-%d %H:%M:%S")

    print(df)
    df = df.round(10)
    df.to_csv(out_f)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-D", action="store_true", help="construct daily submission/dataset using chosen operation.")
    parser.add_argument("-H", action="store_true", help="construct hourly submission/dataset using chosen operation.")
    parser.add_argument("-f", action="store_true", help="filter data using different methods") # TODO: implement parsing sequence of filter operations and construct filter pipline in clean_data
    parser.add_argument("-melt", action="store_true", help="prepare features and melt dataframe into format temporal fusion transformer can accept")
    parser.add_argument("-n", action="store_true", help="normalize the data based on count of households in dwelling.")
    parser.add_argument("-i", type=str, help="specify input file if the operation should run on existing dataset.")
    parser.add_argument("-o", type=str, help="output path and filename")
    parser.add_argument("-s", action="store_true", help="convert input file to submission format in output file.") 
    args = parser.parse_args()

    if args.D and args.H and not args.f:
        parser.print_help()
        raise Exception("choose either the daily or the hourly case. Not both")
    elif not args.D and not args.H and not args.f:
        parser.print_help()
        raise Exception("choose at least one case to predict for. options: [-D, -H]")
    else:
        o_final = None
        if args.f:
            if not args.o and not args.D and args.H:
                o = args.i[:-4] + "_filtered.csv"
            elif not args.o and args.melt:
                o = args.i[:-4] + "_filtered.csv"
                o_final = args.i[:-4] + "_filtered_melted.pkl"
            else:
                o = args.o[:-4] + ".csv"
            clean_data_lerp(args.i, o)
            args.i = o[:-4] + ".csv"
            if not args.D and not args.H:
                exit()
        freq = "h" if args.H else 'D'
        if args.s:
            df = pd.read_csv(args.i)
            make_submission(df, freq, args.o)
        elif args.melt:
            pickle_prepared(prepare_features(get_dataframe('h', in_f=args.i), normalize=args.n), out_f=o_final or o)
