from dataclasses import dataclass, field
from typing import Optional

from darts.models.forecasting.torch_forecasting_model import ForecastingModel

from darts.models import (
    TransformerModel,
    DLinearModel,
    NHiTSModel,
    LightGBMModel,
    XGBModel,
    TFTModel
)

from torch.nn import MSELoss

from .itransformer import ITransformerModel
from .dlinear_film import DLinearFiLMModel

@dataclass
class Model:
    model: ForecastingModel
    params: dict = field(default_factory=dict)

model_map = {
    'tft': Model(
        model=TFTModel,
        params={
            'input_chunk_length': None,
            'output_chunk_length': None,
            'hidden_size': 64,
            'lstm_layers': 1,
            'num_attention_heads': 4,
            'dropout': 0.1,
            'batch_size': 16,
            'n_epochs': 300,
            'add_relative_index': False,
            'loss_fn': MSELoss(),
        }
    ),
    'transformer': Model(
        model=TransformerModel,
        params={ # {'d_model': 2, 'nhead': 24, 'num_encoder_layers': 2, 'num_decoder_layers': 4, 'dim_feedforward': 91, 'dropout': 0.06884646874404202}
            'input_chunk_length': None,
            'output_chunk_length': None,
            'd_model': 24*2,
            'nhead': 24,
            'num_encoder_layers': 2,
            'num_decoder_layers': 4,
            'dim_feedforward': 91,
            'dropout': 0.068846,
            'activation': 'relu',
            # 'norm_type': None,
            # 'use_reversible_instance_norm': True
        }
    ),
    'itransformer': Model(
        model=ITransformerModel,
        params={
            'input_chunk_length': None,
            'output_chunk_length': None,
            'd_model': 64*2,
            'nhead': 32,
            'num_encoder_layers': 2,
            'dim_feedforward': 128,
            'dropout': 0.1,
            'activation': 'relu',
            # 'norm_type': None,
            # 'use_reversible_instance_norm': True
        }
    ),
    'dlinear': Model(
        model=DLinearModel,
        params={
            'input_chunk_length': None,
            'output_chunk_length': None,
        }
    ),
    'dlinear_film': Model(
        model=DLinearFiLMModel,
        params={
            'input_chunk_length': None,
            'output_chunk_length': None,
        }
    ),
    'nhits': Model(
        model=NHiTSModel,
        params = {
            'input_chunk_length': None,
            'output_chunk_length': None,
            'num_stacks': 10,
            'num_blocks': 1,
            'num_layers': 4,
            'layer_widths': 512,
            'pooling_kernel_sizes': None,
            'n_freq_downsample': None,
            'dropout': 0.1,
            'activation': 'ReLU',
            'MaxPool1d': True
        }
    ),
    'lightgbm': Model(
        model=LightGBMModel,
        params = {
            'input_chunk_length': None,
            'lags': None,
            'lags_past_covariates': None,
            'output_chunk_length': None,
            'num_leaves': 31,
            'max_depth': -1,
            'learning_rate': 0.1,
            'n_estimators': 40,
            'min_split_gain': 0,
            'min_child_samples': 0,
            'subsample': 0.6,
            'reg_lambda': 0.5,
            'n_jobs': 16
        }
    ),
    'xgboost': Model(
        model=XGBModel,
        params = {
            'input_chunk_length': None,
            'lags': None,
            'lags_past_covariates': None,
            'output_chunk_length': None,
            'nthread': 16,
            'subsample': 0.6,
            'min_child_weight': 1,
            'n_estimators': 5,
            'max_depth': 4,
            'gamma': 0,
            'eta': 0.1,
            'lambda': 1.0,
            'early_stopping_rounds': 5,
            'verbosity': 0
        }
    )
}

model_tune_map = {
    'transformer': Model(
        model=TransformerModel,
        params={
            'input_chunk_length': None,
            'output_chunk_length': None,
            'd_model': ['int', 1, 4],
            'nhead': ['log_int', 8, 128],
            'num_encoder_layers': ['int', 1, 4],
            'num_decoder_layers': ['int', 1, 4],
            'dim_feedforward': ['log_int', 32, 512],
            'dropout': ['float', 0.01, 0.2],
            'activation': 'relu',
            # 'norm_type': None,
            # 'use_reversible_instance_norm': True
        }
    ),
    'itransformer': Model(
        model=ITransformerModel,
        params={
            'input_chunk_length': None,
            'output_chunk_length': None,
            'd_model': ['int', 1, 4],
            'nhead': ['log_int', 8, 128],
            'num_encoder_layers': ['int', 1, 4],
            'dim_feedforward': ['log_int', 32, 512],
            'dropout': ['float', 0.01, 0.2],
            'activation': 'relu',
            # 'norm_type': None,
            # 'use_reversible_instance_norm': True
        }
    ),
    'dlinear': Model(
        model=DLinearModel,
        params={
            'input_chunk_length': None,
            'output_chunk_length': None,
        }
    ),
    'nhits': Model(
        model=NHiTSModel,
        params = {
            'input_chunk_length': None,
            'output_chunk_length': None,
            'num_stacks': ['int', 2, 15],
            'num_blocks': ['int', 1, 4],
            'num_layers': ['int', 1, 8],
            'layer_widths': ['log_int', 32, 512],
            'pooling_kernel_sizes': None,
            'n_freq_downsample': None,
            'dropout': ['float', 0.01, 0.2],
            'activation': 'ReLU',
            'MaxPool1d': True
        }
    ),
    'lightgbm': Model(
        model=LightGBMModel,
        params = {
            'input_chunk_length': None,
            'lags': None,
            'lags_past_covariates': None,
            'output_chunk_length': None,
            'num_leaves': 31,
            'max_depth': -1,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'min_split_gain': 0,
            'min_child_samples': 0,
            'subsample': 0.6,
            'reg_lambda': 0,
            'n_jobs': 16
        }
    ),
    'xgboost': Model(
        model=XGBModel,
        params = {
            'input_chunk_length': None,
            'lags': None,
            'lags_past_covariates': None,
            'output_chunk_length': None,
            'nthread': 16,
            'subsample': 0.6,
            'min_child_weight': 1,
            'n_estimators': 100,
            'max_depth': 6,
            'gamma': 0,
            'eta': 0.3,
            'lambda': 1.0,
            'early_stopping_rounds': 5
        }
    )
}


def update_dict_A_with_B(A, B):
    B = vars(B)
    for key in A:
        if key in B:
            A[key] = B[key]
    return A

def add_selected_to_A(A, B, selected):
    B = vars(B)
    for key in selected:
        if key in B:
            A[key] = B[key]
    return A

def get_tunable_model(args):
    model_meta = model_tune_map[args.model]

    params = update_dict_A_with_B(model_meta.params, args)
    params = add_selected_to_A(params, args, ['batch_size', 'n_epochs', 'work_dir', 'save_checkpoints', 'model_name', 'log_tensorboard', 'add_encoders', 'pl_trainer_kwargs'])

    if args.model in ['xgboost', 'lightgbm']:
        params['lags'] = params['lags_past_covariates'] = params.pop('input_chunk_length')
    
    print(params)

    return model_meta.model, params

def get_model(args):
    model_meta = model_map[args.model]

    params = update_dict_A_with_B(model_meta.params, args)

    if args.model in ['xgboost', 'lightgbm']:
        params['lags'] = params['lags_past_covariates'] =params.pop('input_chunk_length')
    else:
        params = add_selected_to_A(params, args, ['batch_size', 'n_epochs', 'work_dir', 'save_checkpoints', 'model_name', 'log_tensorboard', 'add_encoders', 'pl_trainer_kwargs'])

    print(params)
    model = model_meta.model(**params)

    return model