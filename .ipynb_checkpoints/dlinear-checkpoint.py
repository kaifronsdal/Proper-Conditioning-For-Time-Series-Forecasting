# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

from typing import Tuple, Optional

import torch
from torch import nn

from gluonts.core.component import validated
from gluonts.model import Input, InputSpec
from gluonts.torch.distributions import StudentTOutput
from gluonts.torch.scaler import StdScaler, MeanScaler, NOPScaler
from gluonts.torch.model.simple_feedforward import make_linear_layer
from gluonts.torch.util import lagged_sequence_values

class MovingAvg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(
            kernel_size=kernel_size, stride=stride, padding=0
        )

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, ...].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, ...].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class SeriesDecomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) Layer.

    Parameters
    ----------
    input_dimension
        Size of incoming feature dimension.
    output_dimension
        Size of output feature dimension.
    """
    def __init__(
        self,
        input_dimension: int,
        output_dimension: int
    ) -> None:
        super().__init__()

        self.gamma_layer = make_linear_layer(input_dimension, output_dimension)
        self.beta_layer = make_linear_layer(input_dimension, output_dimension)

    def forward(self, x: torch.Tensor, conditional: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma_layer(conditional)
        beta = self.beta_layer(conditional)

        return x * (1 + gamma) + beta


class DLinearModel(nn.Module):
    """
    Module implementing a feed-forward model form the paper
    https://arxiv.org/pdf/2205.13504.pdf extended for probabilistic forecasting.

    Parameters
    ----------
    prediction_length
        Number of time points to predict.
    context_length
        Number of time steps prior to prediction time that the model.
    hidden_dimension
        Size of last hidden layers in the feed-forward network.
    distr_output
        Distribution to use to evaluate observations and sample predictions.
    """

    @validated()
    def __init__(
        self,
        prediction_length: int,
        input_dimension: int,
        context_length: int,
        masked_length: int,
        hidden_dimension: int,
        distr_output=StudentTOutput(),
        # lags_seq: Optional[List[int]] = None,
        kernel_size: int = 25,
        scaling: str = "mean",
    ) -> None:
        super().__init__()

        assert prediction_length > 0
        assert context_length > 0

        self.prediction_length = prediction_length
        self.input_dimension = input_dimension
        self.context_length = context_length
        self.masked_length = masked_length

        # self.lags_seq = lags_seq or [0]
        self.past_length = self.context_length# + max(self.lags_seq)
        
        self.hidden_dimension = hidden_dimension
        self.decomposition = SeriesDecomp(kernel_size)

        self.distr_output = distr_output
        if scaling == "mean":
            self.scaler = MeanScaler(keepdim=True)
        elif scaling == "std":
            self.scaler = StdScaler(keepdim=True)
        else:
            self.scaler = NOPScaler(keepdim=True)

        self.kernel_size = kernel_size

        self.linear_seasonal = make_linear_layer(
            masked_length, prediction_length * hidden_dimension
        )
        self.linear_trend = make_linear_layer(
            masked_length, prediction_length * hidden_dimension
        )

        # self.linear_cov = make_linear_layer(
        #     input_dimension * context_length, prediction_length * hidden_dimension
        # )
        # self.FiLM_seasonal = FiLM(
        #     input_dimension * context_length, prediction_length * hidden_dimension
        # )

        # self.FiLM_trend = FiLM(
        #     input_dimension * context_length, prediction_length * hidden_dimension
        # )

        self.args_proj = self.distr_output.get_args_proj(hidden_dimension)

    def describe_inputs(self, batch_size=1) -> InputSpec:
        return InputSpec(
            {
                "past_target": Input(
                    shape=(batch_size, self.past_length), dtype=torch.float
                ),
                "past_observed_values": Input(
                    shape=(batch_size, self.past_length), dtype=torch.float
                ),
                "past_time_feat": Input(
                    shape=(batch_size, self.past_length, self.input_dimension), dtype=torch.float
                ),
            },
            torch.zeros,
        )

    def forward(
        self,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        past_time_feat: torch.Tensor,
    ) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor]:

        ranges = [(1, 217), (312, 385), (480, 553), (648, 769)]

        # input = past_target[..., -self.context_length :]
        # # observed_context = past_observed_values[..., -self.context_length :]

        # prior_input = past_target[..., : -self.context_length]

        # lags = lagged_sequence_values(
        #     self.lags_seq, prior_input, input, dim=-1
        # )

        # input = lags.reshape(lags.shape[0], -1)

        # scale the input
        past_target_scaled, loc, scale = self.scaler(
            past_target, past_observed_values
        )
        
        lags = []

        for r in ranges:
            lags.append(past_target_scaled[:, -r[1]:-r[0]])

        past_target_scaled = torch.concat(lags, dim=-1)
        
        res, trend = self.decomposition(past_target_scaled.unsqueeze(-1))
        seasonal_output = self.linear_seasonal(res.squeeze(-1))
        trend_output = self.linear_trend(trend.squeeze(-1))

        # cov_output = self.linear_cov(past_time_feat.reshape(-1, self.context_length * self.input_dimension))
        
        # nn_out = seasonal_output + trend_output + cov_output
        # cov_feat = past_time_feat.reshape(-1, self.context_length * self.input_dimension)
        
        # nn_out = self.FiLM_seasonal(seasonal_output, cov_feat) + trend_output # self.FiLM_trend(trend_output, cov_feat)
        nn_out = seasonal_output + trend_output

        distr_args = self.args_proj(
            nn_out.reshape(-1, self.prediction_length, self.hidden_dimension)
        )
        return distr_args, loc, scale

# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

import lightning.pytorch as pl
import torch

from gluonts.core.component import validated
from gluonts.torch.modules.loss import DistributionLoss, NegativeLogLikelihood


class DLinearLightningModule(pl.LightningModule):
    """
    A ``pl.LightningModule`` class that can be used to train a
    ``DLinearModel`` with PyTorch Lightning.

    This is a thin layer around a (wrapped) ``DLinearModel`` object,
    that exposes the methods to evaluate training and validation loss.

    Parameters
    ----------
    model_kwargs
        Keyword arguments to construct the ``DLinearModel`` to be trained.
    loss
        Loss function to be used for training.
    lr
        Learning rate.
    weight_decay
        Weight decay regularization parameter.
    """

    @validated()
    def __init__(
        self,
        model_kwargs: dict,
        loss: DistributionLoss = NegativeLogLikelihood(),
        lr: float = 1e-3,
        weight_decay: float = 1e-8,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = DLinearModel(**model_kwargs)
        self.loss = loss
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

    def _compute_loss(self, batch):
        context = batch["past_target"]
        past_observed_values = batch["past_observed_values"]
        past_time_feat = batch["past_time_feat"]
        target = batch["future_target"]
        observed_target = batch["future_observed_values"]

        # print(context.shape, self.model.context_length, self.model.past_length)
        # print(past_time_feat.shape)
        # print(target.shape)
        # print(self.model.describe_inputs())
        assert context.shape[-1] == self.model.context_length
        assert list(past_time_feat.shape[-2:]) == [self.model.context_length, self.model.input_dimension]
        assert target.shape[-1] == self.model.prediction_length

        distr_args, loc, scale = self.model(
            context, past_observed_values=past_observed_values, past_time_feat=past_time_feat
        )
        distr = self.model.distr_output.distribution(distr_args, loc, scale)

        # Initialize the regularization loss
        l1_reg_loss = 0.0
        # The Lambda parameter for L1 Regularization
        l1_lambda = 0.0001 
    
        # Calculate L1 regularization loss
        for param in self.model.parameters():
            l1_reg_loss += torch.norm(param, 1)

        return (
            self.loss(distr, target) * observed_target
        ).sum() / observed_target.sum().clamp_min(1.0) + l1_lambda * l1_reg_loss

    def training_step(self, batch, batch_idx: int):  # type: ignore
        """
        Execute training step.
        """
        train_loss = self._compute_loss(batch)
        self.log(
            "train_loss",
            train_loss,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )
        return train_loss

    def validation_step(self, batch, batch_idx: int):  # type: ignore
        """
        Execute validation step.
        """
        val_loss = self._compute_loss(batch)
        self.log(
            "val_loss", val_loss, on_epoch=True, on_step=False, prog_bar=True
        )
        return val_loss

    def configure_optimizers(self):
        """
        Returns the optimizer to use.
        """
        return torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

from typing import Optional, Iterable, Dict, Any

import torch
import lightning.pytorch as pl

from gluonts.core.component import validated
from gluonts.dataset.common import Dataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.loader import as_stacked_batches
from gluonts.itertools import Cyclic
from gluonts.model.forecast_generator import DistributionForecastGenerator
from gluonts.torch.modules.loss import DistributionLoss, NegativeLogLikelihood
from gluonts.transform import (
    Transformation,
    AddObservedValuesIndicator,
    InstanceSampler,
    InstanceSplitter,
    ValidationSplitSampler,
    TestSplitSampler,
    ExpectedNumInstanceSampler,
    SelectFields,
)
from gluonts.torch.model.estimator import PyTorchLightningEstimator
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.torch.distributions import (
    DistributionOutput,
    StudentTOutput,
)

PREDICTION_INPUT_NAMES = ["past_target", "past_observed_values", "past_time_feat"]

TRAINING_INPUT_NAMES = PREDICTION_INPUT_NAMES + [
    "future_target",
    "future_observed_values",
    "future_time_feat",
]


class DLinearEstimator(PyTorchLightningEstimator):
    """
    An estimator training the d-linear model form the paper
    https://arxiv.org/pdf/2205.13504.pdf extended for probabilistic forecasting.

    This class is uses the model defined in ``DLinearModel``,
    and wraps it into a ``DLinearLightningModule`` for training
    purposes: training is performed using PyTorch Lightning's ``pl.Trainer``
    class.

    Parameters
    ----------
    prediction_length
        Length of the prediction horizon.
    context_length
        Number of time steps prior to prediction time that the model
        takes as inputs (default: ``10 * prediction_length``).
    hidden_dimension
        Size of representation.
    lr
        Learning rate (default: ``1e-3``).
    weight_decay
        Weight decay regularization parameter (default: ``1e-8``).

    distr_output
        Distribution to use to evaluate observations and sample predictions
        (default: StudentTOutput()).
    loss
        Loss to be optimized during training
        (default: ``NegativeLogLikelihood()``).
    kernel_size
    batch_size
        The size of the batches to be used for training (default: 32).
    num_batches_per_epoch
        Number of batches to be processed in each training epoch
            (default: 50).
    trainer_kwargs
        Additional arguments to provide to ``pl.Trainer`` for construction.
    train_sampler
        Controls the sampling of windows during training.
    validation_sampler
        Controls the sampling of windows during validation.
    """

    @validated()
    def __init__(
        self,
        prediction_length: int,
        input_dimension: int,
        context_length: Optional[int] = None,
        masked_length: Optional[int] = None,
        hidden_dimension: Optional[int] = None,
        lags_sequence: Optional[int] = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-8,
        scaling: Optional[str] = "mean",
        distr_output: DistributionOutput = StudentTOutput(),
        loss: DistributionLoss = NegativeLogLikelihood(),
        kernel_size: int = 25,
        batch_size: int = 32,
        num_batches_per_epoch: int = 50,
        trainer_kwargs: Optional[Dict[str, Any]] = None,
        train_sampler: Optional[InstanceSampler] = None,
        validation_sampler: Optional[InstanceSampler] = None,
    ) -> None:
        default_trainer_kwargs = {
            "max_epochs": 100,
            "gradient_clip_val": 10.0,
        }
        if trainer_kwargs is not None:
            default_trainer_kwargs.update(trainer_kwargs)
        super().__init__(trainer_kwargs=default_trainer_kwargs)

        self.prediction_length = prediction_length
        self.context_length = context_length or 10 * prediction_length
        self.masked_length = masked_length or context_length
        self.lags_sequence = lags_sequence or 0
        # TODO find way to enforce same defaults to network and estimator
        # somehow
        self.hidden_dimension = hidden_dimension or 20
        self.input_dimension = input_dimension
        self.lr = lr
        self.weight_decay = weight_decay
        self.distr_output = distr_output
        self.scaling = scaling
        self.loss = loss
        self.kernel_size = kernel_size
        self.batch_size = batch_size
        self.num_batches_per_epoch = num_batches_per_epoch

        self.train_sampler = train_sampler or ExpectedNumInstanceSampler(
            num_instances=1.0, min_future=prediction_length
        )
        self.validation_sampler = validation_sampler or ValidationSplitSampler(
            min_future=prediction_length
        )

    def create_transformation(self) -> Transformation:
        # return SelectFields(
        #     [
        #         FieldName.ITEM_ID,
        #         FieldName.INFO,
        #         FieldName.START,
        #         FieldName.TARGET,
        #         FieldName.FEAT_TIME
        #     ],
        #     allow_missing=True,
        # )
        return AddObservedValuesIndicator(
            target_field=FieldName.TARGET,
            output_field=FieldName.OBSERVED_VALUES,
        )

    def create_lightning_module(self) -> pl.LightningModule:
        return DLinearLightningModule(
            loss=self.loss,
            lr=self.lr,
            weight_decay=self.weight_decay,
            model_kwargs={
                "prediction_length": self.prediction_length,
                "context_length": self.context_length,# + self.lags_sequence,
                "masked_length": self.masked_length,
                "input_dimension": self.input_dimension,
                "hidden_dimension": self.hidden_dimension,
                "distr_output": self.distr_output,
                "kernel_size": self.kernel_size,
                "scaling": self.scaling,
            },
        )

    def _create_instance_splitter(
        self, module: DLinearLightningModule, mode: str
    ):
        assert mode in ["training", "validation", "test"]

        instance_sampler = {
            "training": self.train_sampler,
            "validation": self.validation_sampler,
            "test": TestSplitSampler(),
        }[mode]

        return InstanceSplitter(
            target_field=FieldName.TARGET,
            is_pad_field=FieldName.IS_PAD,
            start_field=FieldName.START,
            forecast_start_field=FieldName.FORECAST_START,
            instance_sampler=instance_sampler,
            past_length=self.context_length,
            future_length=self.prediction_length,
            time_series_fields=[
                FieldName.OBSERVED_VALUES,
                FieldName.FEAT_TIME
            ],
            dummy_value=self.distr_output.value_in_support,
        )

    def create_training_data_loader(
        self,
        data: Dataset,
        module: DLinearLightningModule,
        shuffle_buffer_length: Optional[int] = None,
        **kwargs,
    ) -> Iterable:
        data = Cyclic(data).stream()
        instances = self._create_instance_splitter(module, "training").apply(
            data, is_train=True
        )
        return as_stacked_batches(
            instances,
            batch_size=self.batch_size,
            shuffle_buffer_length=shuffle_buffer_length,
            field_names=TRAINING_INPUT_NAMES,
            output_type=torch.tensor,
            num_batches_per_epoch=self.num_batches_per_epoch,
        )

    def create_validation_data_loader(
        self,
        data: Dataset,
        module: DLinearLightningModule,
        **kwargs,
    ) -> Iterable:
        instances = self._create_instance_splitter(module, "validation").apply(
            data, is_train=True
        )
        return as_stacked_batches(
            instances,
            batch_size=self.batch_size,
            field_names=TRAINING_INPUT_NAMES,
            output_type=torch.tensor,
        )

    def create_predictor(
        self,
        transformation: Transformation,
        module,
    ) -> PyTorchPredictor:
        prediction_splitter = self._create_instance_splitter(module, "test")

        return PyTorchPredictor(
            input_transform=transformation + prediction_splitter,
            input_names=PREDICTION_INPUT_NAMES,
            prediction_net=module,
            forecast_generator=DistributionForecastGenerator(
                self.distr_output
            ),
            batch_size=self.batch_size,
            prediction_length=self.prediction_length,
            device="auto",
        )