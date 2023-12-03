import argparse

import torch
import torch.nn as nn
import torch.optim as optim
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
from darts.metrics import mape
from darts.utils.statistics import check_seasonality, plot_acf
from darts.datasets import AirPassengersDataset, SunspotsDataset

from data.load import get_data, split_timeseries
from models.load import get_model

from lightning.pytorch import seed_everything

import os

# import warnings

# warnings.filterwarnings("ignore")
# import logging

# logging.disable(logging.CRITICAL)

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
    parser.add_argument('--save_checkpoints', type=bool, default=False)
    
    # forecasting task
    parser.add_argument('--input_chunk_length', type=int, default=12, help='Number of time steps in the past to take as a model input (per chunk). Applies to the target series, and past and/or future covariates (if the model supports it).')
    parser.add_argument('--output_chunk_length', type=int, default=1, help='Number of time steps predicted at once (per chunk) by the internal model.')
    parser.add_argument('--d_model', type=int, default=16, help='The number of expected features in the transformer encoder/decoder inputs (default=64).')

    # training
    parser.add_argument('--batch_size', type=int, default=64, help='Bumber of series in each trainng pass.')
    parser.add_argument('--n_epochs', type=int, default=100, help='Number of epochs to train for.')

    args = parser.parse_args()

    args.model_name = f"{args.model}_{args.data}_{datetime.now().strftime('%d-%m-%Y_%H:%M:%S')}"

    args.pl_trainer_kwargs = {"accelerator": "gpu", "devices": [args.device]}
    seed_everything(42, workers=True)
    # os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"]=str(args.device)

    data, split, add_encoders, scaling = get_data(args.data, args.cache_path)

    args.add_encoders = add_encoders

    model = get_model(args)

    print(model)

    data_split = split_timeseries(data, split)
    if len(data_split) == 3:
        train, valid, test = data_split
    elif len(data_split) == 2:
        train, test = data_split
        valid = None
    else:
        raise ValueError(f"Expected 2 or 3 splits for data, but found {len(data_split)}.")

    model.fit(train, val_series=valid)
    
