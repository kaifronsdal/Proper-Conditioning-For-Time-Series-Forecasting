import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics import MetricCollection

import numpy as np
import pandas as pd
import shutil
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm_notebook as tqdm

from datetime import datetime

from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import TransformerModel, ExponentialSmoothing
import darts.metrics as metrics
from darts.utils.statistics import check_seasonality, plot_acf
from darts.datasets import AirPassengersDataset, SunspotsDataset

from data.load import get_data, split_timeseries
from models.load import get_model, get_tunable_model

from lightning.pytorch import seed_everything
from pytorch_lightning.callbacks import Callback
from optuna.integration import PyTorchLightningPruningCallback

from darts.dataprocessing.encoders import SequentialEncoder

from pytorch_lightning.callbacks import EarlyStopping

import optuna

import os

torch.set_num_threads(5)
torch.set_num_interop_threads(5)

# class ComputeMetrics:
#     def __init__(self, prefix):
#         self.prefix = prefix

#     def __len__(self):
#         return True
    
#     def __call__(self, output, target):
#         return {f'{self.prefix}_mae': metrics.mae(output, target)}

# class FakeMetricCollection(MetricCollection):
#     def clone(self, prefix):
#         print('==================')
#         return ComputeMetrics(prefix)

# class MetricsCallback(Callback):
#     def setup(self, trainer, pl_module, stage):
#         print('====================')
#         print(pl_module)
#         breakpoint()
#         pl_module.val_metrics = compute_metrics
#         pl_module.train_metrics = compute_metrics

# import warnings

# warnings.filterwarnings("ignore")
# import logging

def train_model(args, train, train_cov, valid, valid_cov, test, inference_cov, metric_list):
    args.model_name = f"{args.model}_{args.data}_{datetime.now().strftime('%d-%m-%Y_%H:%M:%S')}"
    
    model = get_model(args)
    print(model)

    # breakpoint()
    # breakpoint()
    model.fit(train, past_covariates=train_cov, val_series=valid, val_past_covariates=valid_cov)

    if hasattr(type(model), 'load_from_checkpoint'):
        model = type(model).load_from_checkpoint(model.model_name, '/lfs/turing3/0/kaif/GitHub/ts/checkpoints')
    
    results = {name: [] for name in metric_list.keys()}
    
    pred_series = model.predict(n=n, series=valid, past_covariates=inference_cov)
    
    for p, v, t in zip(pred_series, valid, test):
        for name, m in metric_list.items():
            if name == 'mase':
                results[name].append(m(t, p, v))
            else:
                results[name].append(m(t, p))
    
    for name in metric_list.keys():
        print(f'{name}: {np.mean(results[name])}')
        if hasattr(model, 'trainer'):
            model.trainer.logger.log_metrics({name: np.mean(results[name])})

    for i in range(min(5, len(train))):
        # train[i].plot()
        valid[i].plot()
        test[i].plot(label="actual")
        pred_series[i].plot(label="forecast")
        plt.savefig(f'plt_{args.model}_{i}.png')
        plt.clf()
    
    return {k: np.mean(v) for k, v in results.items()}


# logging.disable(logging.CRITICAL)
def objective(model_cls, config, train, valid, test):
    def _objective(trial):
        # callback = [PyTorchLightningPruningCallback(trial, monitor="val_loss")]
        callback = [EarlyStopping(
            monitor="val_loss",
            patience=5,
            # min_delta=0.05,
            mode='min',
        )]
    
        # # set input_chunk_length, between 5 and 14 days
        # days_in = trial.suggest_int("days_in", 5, 14)
        # in_len = days_in * DAY_DURATION
    
        # # set out_len, between 1 and 13 days (it has to be strictly shorter than in_len).
        # days_out = trial.suggest_int("days_out", 1, days_in - 1)
        # out_len = days_out * DAY_DURATION
    
        # Other hyperparameters
        # kernel_size = trial.suggest_int("kernel_size", 5, 25)
        # num_filters = trial.suggest_int("num_filters", 5, 25)
        # weight_norm = trial.suggest_categorical("weight_norm", [False, True])
        # dilation_base = trial.suggest_int("dilation_base", 2, 4)
        # dropout = trial.suggest_float("dropout", 0.0, 0.4)
        # lr = trial.suggest_float("lr", 5e-5, 1e-3, log=True)
        # include_dayofweek = trial.suggest_categorical("dayofweek", [False, True])
        params = {}
        for name, sug in config.items():
            if not isinstance(sug, list):
                params[name] = sug
            else:
                kind = sug[0]
                range = sug[1:]
                if kind == 'int':
                    params[name] = trial.suggest_int(name, *range)
                elif kind == 'log_int':
                    params[name] = trial.suggest_int(name, *range, log=True)
                elif kind == 'float':
                    params[name] = trial.suggest_float(name, *range)
                else:
                    raise ValueError("Unkown optuna suggestion kind.")

        if 'nhead' in params:
            if params['nhead'] % 2 == 1:
                params['nhead'] -= 1
        if 'd_model' in params:
            params['d_model'] = params['d_model'] * params['nhead']

        params['optimizer_kwargs'] = {'lr': trial.suggest_float('lr', 0.00001, 0.001, log=True)}
        
        # build and train the TCN model with these hyper-parameters:
        params['pl_trainer_kwargs']['callbacks'] = callback
        model = model_cls(**params)
    
        model.fit(train, val_series=valid)

        metric_list = {
            # 'mape': metrics.mape,
            'mae': metrics.mae,
            'mse': metrics.mse,
            'mase': metrics.mase
        }
    
        results = {name: [] for name in metric_list.keys()}
        
        pred_series = model.predict(n=len(test), series=valid)
        
        for p, v, t in zip(pred_series, valid, test):
            for name, m in metric_list.items():
                if name == 'mase':
                    results[name].append(m(t, p, v))
                else:
                    results[name].append(m(t, p))

        return np.mean(results['mse'])
    
    return _objective

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Model')

    # dataset
    parser.add_argument('--data', type=str, required=True, default='ettm1', help='dataset type')
    parser.add_argument('--cache_path', type=str, default='/lfs/turing3/0/kaif/GitHub/ts/data/cache', help='dataset cache location')

    parser.add_argument('--work_dir', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--device', type=int, default=0, help='gpu to use')

    # model
    parser.add_argument('--model', type=str, required=True, default='Transformer',
                        help='model name, options: [Autoformer, Transformer, DLinear]')
    parser.add_argument('--log_tensorboard', type=bool, default=True)
    parser.add_argument('--save_checkpoints', type=bool, default=True)
    
    # forecasting task
    parser.add_argument('--input_chunk_length', type=int, default=96, help='Number of time steps in the past to take as a model input (per chunk). Applies to the target series, and past and/or future covariates (if the model supports it).')
    parser.add_argument('--output_chunk_length', type=int, default=96, help='Number of time steps predicted at once (per chunk) by the internal model.')
    # parser.add_argument('--d_model', type=int, default=64, help='The number of expected features in the transformer encoder/decoder inputs (default=64).')

    # training
    parser.add_argument('--batch_size', type=int, default=32, help='Number of series in each trainng pass.')
    parser.add_argument('--n_epochs', type=int, default=8, help='Number of epochs to train for.')

    args = parser.parse_args()

    args.model_name = f"{args.model}_{args.data}_{datetime.now().strftime('%d-%m-%Y_%H:%M:%S')}"
    args.optimizer_kwargs = {'lr': 0.001}
    args.pl_trainer_kwargs = {
        "accelerator": "gpu", "devices": [args.device],
        "limit_train_batches": 300,
        "limit_val_batches": 200,
        # "callbacks": [EarlyStopping(
        #     monitor="val_loss",
        #     patience=5,
        #     # min_delta=0.05,
        #     mode='min',
        # )]
        # "enable_progress_bar": False
    }
    # args.torch_metrics = FakeMetricCollection([])
    
    seed_everything(42, workers=True)
    # os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"]=str(args.device)

    data, split, add_encoders, scaling = get_data(args.data, args.cache_path)

    # ============
    # data_split = split_timeseries(data, split)
    # if len(data_split) == 3:
    #     train, valid, test = data_split
    # elif len(data_split) == 2:
    #     train, test = data_split
    #     valid = None
    # else:
    #     raise ValueError(f"Expected 2 or 3 splits for data, but found {len(data_split)}.")

    # # args.add_encoders = add_encoders

    # ms = ['transformer', 'itransformer', 'nhits']
    # results = {}
    # for m in ms:
    #     args.model = m
    #     model_cls, params = get_tunable_model(args)
    
    #     obj = objective(model_cls, params, train, valid, test)
    
    #     def print_callback(study, trial):
    #         print(f"Current value: {trial.value}, Current params: {trial.params}")
    #         print(f"Best value: {study.best_value}, Best params: {study.best_trial.params}")
        
    #     study_name = f"{m}-{args.data}-mse-study"  # Unique identifier of the study.
    #     storage_name = "sqlite:///{}.db".format(study_name)
    #     study = optuna.create_study(direction="minimize", study_name=study_name, storage=storage_name)
    #     # study = optuna.create_study(direction="minimize")
        
    #     study.optimize(obj, timeout=60*30, callbacks=[print_callback])
        
    #     # We could also have used a command as follows to limit the number of trials instead:
    #     # study.optimize(objective, n_trials=100, callbacks=[print_callback])
        
    #     # Finally, print the best value and best hyperparameters:
    #     print(f"Best value: {study.best_value}, Best params: {study.best_trial.params}")
    #     results[m] = study

    # breakpoint()
    # ============

    NR_DAYS = 80
    DAY_DURATION = 24 * 4  # 15 minutes frequency
    
    data = [d[-NR_DAYS*DAY_DURATION:] for d in data]
    
    data_split = split_timeseries(data, split)
    if len(data_split) == 4:
        _, train, valid, test = data_split
    elif len(data_split) == 3:
        train, valid, test = data_split
    elif len(data_split) == 2:
        train, test = data_split
        valid = None
    else:
        raise ValueError(f"Expected 2 or 3 splits for data, but found {len(data_split)}.")

    # breakpoint()
    
    seq_encoder = SequentialEncoder(
        add_encoders=add_encoders,
        input_chunk_length=args.input_chunk_length,
        output_chunk_length=args.output_chunk_length,
        takes_past_covariates=True,
        takes_future_covariates=False,
        lags_past_covariates=None,
        lags_future_covariates=None,
    )

    # Fit and transform train_series
    train_cov, _ = seq_encoder.encode_train(target=train)

    # breakpoint()
    
    # # Transform valid_series and test_series 
    valid_cov, _ = seq_encoder.encode_train(target=valid) if valid is not None else None
    test_cov, _ = seq_encoder.encode_train(target=test)

    n = 96 * 2
    inference_cov, _ = seq_encoder.encode_inference(
        n=n,
        target=valid,
    )

    metric_list = {
        # 'mape': metrics.mape,
        'mae': metrics.mae,
        'mse': metrics.mse,
        # 'mase': metrics.mase
    }
    
    ms = ['dlinear', 'dlinear_film', 'transformer', 'itransformer', 'nhits', 'xgboost']
    # ms = ['dlinear_film']
    results = {}
    for m in ms:
        args.model = m
        results[m] = train_model(args, train, train_cov, valid, valid_cov, test, inference_cov, metric_list)
    
    print(results)
    breakpoint()
