import gc

from pandas.api.types import is_datetime64_any_dtype as is_datetime
from pandas.api.types import is_categorical_dtype
import plotly.express as px
import numpy as np
import pandas as pd
import plotly.offline as py
import lightgbm as lgb
from tqdm import tqdm_notebook as tqdm
from sklearn.metrics import mean_squared_log_error

py.init_notebook_mode(connected=True)


def lgbm_fit(train, val, seed=None, cat_features=None, num_rounds=1500, lr=0.1, bf=0.1):
    X_train, y_train = train
    X_valid, y_valid = val
    metric = 'root_mean_squared_error'
    params = {
        'num_leaves': 31,
        'force_row_wise': True,
        'objective': 'regression',
        'learning_rate': lr,
        "boosting": "gbdt",
        "bagging_freq": 5,
        "bagging_fraction": bf,
        "feature_fraction": 0.9,
        "metric": metric,
        'seed': seed
    }

    d_train = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features)
    d_valid = lgb.Dataset(X_valid, label=y_valid, categorical_feature=cat_features)
    watchlist = [d_train, d_valid]

    print('training LGB:')
    model = lgb.train(params,
                      train_set=d_train,
                      num_boost_round=num_rounds,
                      categorical_feature=cat_features,
                      valid_sets=watchlist,
                      callbacks=[
                          lgb.early_stopping(stopping_rounds=20, first_metric_only=False, verbose=True),
                          lgb.log_evaluation(period=20, show_stdv=True)
                      ])

    y_pred_valid = model.predict(X_valid, num_iteration=model.best_iteration)

    print('best_score', model.best_score)

    del d_train, d_valid, watchlist
    gc.collect()

    return model, y_pred_valid, y_valid


def reduce_mem_usage(df, use_float16=False):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage: {:.2f} MB'.format(start_mem))

    for col in df.columns:
        if is_datetime(df[col]) or is_categorical_dtype(df[col]):
            continue
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if use_float16 and c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage (after optimization): {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    gc.collect()
    return df


def plot_date_usage(train_df, meter=0, building_id=0):
    train_temp_df = train_df[train_df['meter'] == meter]
    train_temp_df = train_temp_df[train_temp_df['building_id'] == building_id]
    train_temp_df_meter = train_temp_df.groupby('date')['meter_reading_log1p'].sum()
    train_temp_df_meter = train_temp_df_meter.to_frame().reset_index()
    fig = px.line(train_temp_df_meter, x='date', y='meter_reading_log1p')
    fig.show()
    gc.collect()


def lgbm_predict(X_test, models, batch_size=1000000):
    iterations = (X_test.shape[0] + batch_size - 1) // batch_size
    print('iterations', iterations)

    y_test_pred_total = np.zeros(X_test.shape[0])
    for i, model in enumerate(models):
        print(f'predicting {i}-th model')
        for k in tqdm(range(iterations)):
            y_pred_test = model.predict(
                X_test[k * batch_size:(k + 1) * batch_size],
                num_iteration=model.best_iteration
            )
            y_test_pred_total[k * batch_size:(k + 1) * batch_size] += y_pred_test
            del y_pred_test
            gc.collect()

    y_test_pred_total /= len(models)
    gc.collect()
    return y_test_pred_total


def add_lag_feature(df, cols, window=3, groupby=None):
    data = df

    if groupby is not None:
        data = data.groupby(groupby)

    rolled = data[cols].rolling(window=window, min_periods=0)
    lag_mean = rolled.mean().reset_index().astype(np.float16)
    lag_max = rolled.max().reset_index().astype(np.float16)
    lag_min = rolled.min().reset_index().astype(np.float16)
    lag_std = rolled.std().reset_index().astype(np.float16)
    for col in cols:
        df[f'{col}_mean_lag{window}'] = lag_mean[col]
        df[f'{col}_max_lag{window}'] = lag_max[col]
        df[f'{col}_min_lag{window}'] = lag_min[col]
        df[f'{col}_std_lag{window}'] = lag_std[col]
    gc.collect()
    return df


def interpolate(df):
    df__timestamps = df['timestamp']
    df__interpolated = df.loc[:, df.columns != 'timestamp'].groupby(
        'site_id').apply(lambda group: group.interpolate(limit_direction='both'))
    gc.collect()
    return pd.concat([df__timestamps, df__interpolated], axis=1)


def add_temporal_feature(df):
    df["hour"] = df["timestamp"].dt.hour
    df["month"] = df["timestamp"].dt.month
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    gc.collect()
    return df


def add_building_features(df, building_meter_data=None):
    if not building_meter_data:

        df_group = df.groupby('building_id')['meter_reading_log1p']
        building_meter_data = {
            "building_mean": df_group.mean().astype(np.float16),
            "building_median": df_group.median().astype(np.float16),
            "building_min": df_group.min().astype(np.float16),
            "building_max": df_group.max().astype(np.float16),
            "building_std": df_group.std().astype(np.float16)
        }

    df['building_mean'] = df['building_id'].map(building_meter_data["building_mean"])
    df['building_median'] = df['building_id'].map(building_meter_data["building_median"])
    df['building_min'] = df['building_id'].map(building_meter_data["building_min"])
    df['building_max'] = df['building_id'].map(building_meter_data["building_max"])
    df['building_std'] = df['building_id'].map(building_meter_data["building_std"])

    gc.collect()

    return df, building_meter_data
