from dataclasses import dataclass, field
from darts.datasets.dataset_loaders import DatasetLoader
from typing import Optional
from pathlib import Path
import pickle

import numpy as np

from darts.dataprocessing.transformers import Scaler, MissingValuesFiller
from darts.dataprocessing import Pipeline

from darts.datasets import (
    ElectricityDataset,
    AirPassengersDataset,
    EnergyDataset,
    ExchangeRateDataset,
    ILINetDataset,
    TrafficDataset,
    UberTLCDataset,
    WeatherDataset,
    ETTm1Dataset,
    ETTm2Dataset
)
from sklearn.preprocessing import StandardScaler


@dataclass
class Dataset:
    source: DatasetLoader
    split: list
    loader_params: dict = field(default_factory=dict)
    preprocessing: Optional[list] = None
    encoders: dict = field(default_factory=dict)

def get_scaler():
    return Scaler(scaler=StandardScaler())
    # return Scaler()

def get_missing_filler():
    return MissingValuesFiller()

def get_scaler_missing():
    return Pipeline([MissingValuesFiller(), Scaler()])

dataset_map = {
    'electricity': Dataset(
        source=ElectricityDataset,
        split=[0.6, 0.8],
        loader_params={'multivariate': True},
        encoders= {
            'cyclic': {'past': ['month', 'dayofyear', 'dayofweek', 'hour']},
            'datetime_attribute': {'past': ['year', 'month']},
            'transformer': get_scaler(),
        }
    ),
    'exchange': Dataset(
        source=ExchangeRateDataset,
        split=[0.25, 0.92, 0.96],
        loader_params={'multivariate': True},
        encoders= {
            'position': {'past': ['relative']},
            'custom': {'past': [np.cos, np.sin]},
            'transformer': get_scaler(),
        }
    ),
    'ili': Dataset(
        source=ILINetDataset,
        split=[0.6,0.8],
        loader_params={'multivariate': True},
        encoders= {
            'cyclic': {'past': ['month', 'dayofyear', 'dayofweek', 'hour']},
            'datetime_attribute': {'past': ['year', 'month']},
            'transformer': get_scaler_missing(),
        }
    ),
    'traffic': Dataset(
        source=TrafficDataset,
        split=[0.6,0.8],
        loader_params={'multivariate': True},
        encoders= {
            'cyclic': {'past': ['month', 'dayofyear', 'dayofweek', 'hour']},
            'datetime_attribute': {'past': ['year', 'month']},
            'transformer': get_scaler(),
        }
    ),
    'uber': Dataset(
        source=UberTLCDataset,
        split=[0.6,0.8],
        loader_params={'multivariate': True},
        encoders= {
            'cyclic': {'past': ['month', 'dayofyear', 'dayofweek', 'hour']},
            'datetime_attribute': {'past': ['year', 'month']},
            'transformer': get_scaler(),
        }
    ),
    'weather': Dataset(
        source=WeatherDataset,
        split=[0.8, 0.99, 0.995],
        loader_params={'multivariate': True},
        encoders= {
            'cyclic': {'past': ['month', 'dayofyear', 'dayofweek', 'hour']},
            'datetime_attribute': {'past': ['year', 'month']},
            'transformer': get_scaler(),
        }
    ),
    'ettm1': Dataset(
        source=ETTm1Dataset,
        split=[0.6,0.8],
        encoders= {
            'cyclic': {'past': ['month', 'dayofyear', 'dayofweek', 'hour']},
            'datetime_attribute': {'past': ['year', 'month']},
            'transformer': get_scaler(),
        }
    ),
    'ettm2': Dataset(
        source=ETTm2Dataset,
        split=[0.6,0.8],
        encoders= {
            'cyclic': {'past': ['month', 'dayofyear', 'dayofweek', 'hour']},
            'datetime_attribute': {'past': ['year', 'month']},
            'transformer': get_scaler(),
        }
    ),
    'airpassenger': Dataset(
        source=AirPassengersDataset,
        split=[0.6,0.8],
        encoders= {
            'cyclic': {'past': ['month']},
            'datetime_attribute': {'past': ['year']},
            'transformer': get_scaler(),
        }
    ),
    # 'energy': Dataset(
    #     source=EnergyDataset,
    #     split=[0.6,0.8],
    #     encoders= {
    #         'cyclic': {'past': ['month', 'dayofyear', 'dayofweek', 'hour']},
    #         'datetime_attribute': {'past': ['year', 'month']},
    #         'transformer': get_scaler(),
    #     }
    # ),
}


def get_data(dataset: str, cache_path):
    data_meta = dataset_map[dataset]

    cache_path = Path(cache_path) / f"{dataset}.pkl"
    
    if cache_path.exists():  # Data already exists in the cache.
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
    else:  # Data needs to be downloaded/generated.
        data = data_meta.source(**data_meta.loader_params).load()

        # if data_meta.loader_params.get('multivariate', False) and not isinstance(data, list):
        if not isinstance(data, list):
            data = [data[component] for component in data.components]

        if isinstance(data, list):
            for i, d in enumerate(data):
                if np.any(np.isnan(d.values())):
                    filler = MissingValuesFiller()
                    data[i] = filler.transform(d)

        else:
           if np.any(np.isnan(data.values())):
               filler = MissingValuesFiller()
               data = filler.transform(data)
        
        # Saving data to cache.
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        print(cache_path)
        with open(cache_path, "wb") as f:
            pickle.dump(data, f)

    if not isinstance(data, list):
        data = [data]

    mins, maxs = [], []
    for i, d in enumerate(data):
        m = d.min(axis=0).values().item()
        M = d.max(axis=0).values().item()
        mins.append(m)
        maxs.append(M)

        data[i] = (d - m) / (M - m)
        
    return data, data_meta.split, data_meta.encoders, (mins, maxs)



def split_timeseries(data, splits: list):
    last_split = 0
    relative_splits = []
    for s in splits:
        relative_splits.append((s - last_split) / (1 - last_split))
        last_split = s

    split_data = [[] for _ in range(len(splits) + 1)]
    for d in data:
        right = d
        for i, s in enumerate(relative_splits):
            left, right = right.split_after(s)
            split_data[i].append(left)
        split_data[-1].append(right)

    return split_data
    
        

