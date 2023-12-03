from dataclasses import dataclass, field
from typing import Optional

from darts.models.forecasting.torch_forecasting_model import ForecastingModel

from darts.models import (
    TransformerModel,
    DLinearModel,
    NHiTSModel
)

@dataclass
class Model:
    model: ForecastingModel
    params: dict = field(default_factory=dict)


model_map = {
    'transformer': Model(
        model=TransformerModel,
        params={
            'input_chunk_length': None,
            'output_chunk_length': None,
            'd_model': 16,
            'nhead': 8,
            'num_encoder_layers': 2,
            'num_decoder_layers': 2,
            'dim_feedforward': 128,
            'dropout': 0.1,
            'activation': 'relu',
            'norm_type': None
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
            'num_stacks': 3,
            'num_blocks': 1,
            'num_layers': 2,
            'layer_widths': 128,
            'pooling_kernel_sizes': None,
            'n_freq_downsample': None,
            'dropout': 0.1,
            'activation': 'ReLU',
            'MaxPool1d': True
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

def get_model(args):
    model_meta = model_map[args.model]

    params = update_dict_A_with_B(model_meta.params, args)
    params = add_selected_to_A(params, args, ['batch_size', 'n_epochs', 'work_dir', 'save_checkpoints', 'model_name', 'log_tensorboard', 'add_encoders', 'pl_trainer_kwargs'])
    
    model = model_meta.model(**params)

    return model