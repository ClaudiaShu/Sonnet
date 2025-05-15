import json
import numpy as np

import lightning as L
import torch
import torch.nn.functional as F
import functools

from sonnet.utils.metrics import acc
from gluonts.evaluation.metrics import (
    mase,
    mape,
    mse,
    smape,
)
from torch.optim.lr_scheduler import (
    ConstantLR,
    LinearLR,
    CosineAnnealingLR,
)

city_list = ["london", "newyork", "hongkong", "capetown", "singapore"]

# Initialize dataset_target_map with non-city entries
dataset_target_map = {
    "electricity": "Electricity value NE5",
    "energy": "Energy price",
    "etth1": "Oil temperature",
    "etth2": "Oil temperature",
    "eng": "Flu rate",
    "us2": "Flu rate",
    "us9": "Flu rate",
}

# Initialize dataset_title_map with non-city entries
dataset_title_map = {
    "electricity": "Forecasting results on electricity value NE5",
    "energy": "Forecasting results on energy price",
    "etth1": "Forecasting results on oil temperature",
    "etth2": "Forecasting results on oil temperature",
    "eng": "Forecasting results on flu rate in England",
    "us2": "Forecasting results on flu rate in US region 2",
    "us9": "Forecasting results on flu rate in US region 9",
}

# Add all cities to both dictionaries automatically
for city in city_list:
    dataset_target_map[city] = "temperature 500 hPa"
    # Format city name with the first letter capitalized for better presentation
    capitalized_city = " ".join(
        word.capitalize() for word in city.replace("_", " ").split()
    )
    dataset_title_map[city] = (
        f"Forecasting results on temperature 500 hPa in {capitalized_city}"
    )


def make_output_prediction(func):
    @functools.wraps(func)
    def wrapper(self, batch, batch_idx, *args, **kwargs):
        inputs, targets = batch
        outputs = self.forward(inputs)

        if torch.isnan(outputs).any():
            self.stop_on_sequence_error = True
            raise ValueError("NaN detected in outputs")

        return func(self, (targets, outputs), batch_idx, *args, **kwargs)

    return wrapper


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        # Convert any NumPy type to its Python equivalent
        if isinstance(obj, np.generic):
            return obj.item()
        # Optionally convert numpy arrays to list (if not already converted)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class BaseModel(L.LightningModule):
    def __init__(
        self,
        configs,
        datamodule,
        lr: float = 1e-3,
        weight_decay: float = 1e-8,
        lr_scheduler: str = "cosine_lr",
        cosine_annealing_lr_args: dict = {"T_max": 10, "eta_min": 1e-5},
        reverse_eval: bool = False,
    ):
        super().__init__()
        self.configs = configs
        self.datamodule = datamodule
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_scheduler = lr_scheduler
        self.cosine_annealing_lr_args = cosine_annealing_lr_args
        self.reverse_eval = reverse_eval

        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.test_results = []
        self.stop_on_sequence_error = False

    def forward(self, x):
        pass

    @make_output_prediction
    def training_step(self, batch, batch_idx):
        targets, outputs = batch
        loss = F.mse_loss(outputs, targets)
        self.log("train_loss", loss)
        return loss

    @make_output_prediction
    def validation_step(self, batch, batch_idx):
        targets, outputs = batch
        loss = F.mse_loss(outputs, targets)
        tar_loss = F.mse_loss(outputs[:, -1, :], targets[:, -1, :])

        if not hasattr(self, "validation_step_outputs"):
            self.validation_step_outputs = []
        if self.stop_on_sequence_error:
            self.validation_step_outputs.append(loss)
        else:
            self.validation_step_outputs.append(tar_loss)

        values = {
            "val_seq_loss": loss,
            "val_tar_loss": tar_loss,
        }  # add more items if needed
        self.log_dict(values)
        return loss

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        avg_loss = sum(outputs) / len(outputs)
        self.log("val_loss", avg_loss)

    @make_output_prediction
    def test_step(self, batch, batch_idx):
        targets, outputs = batch

        # detach
        targets = targets.cpu().numpy()
        outputs = outputs.cpu().numpy()

        # Inverse transform the outputs and targets
        if self.datamodule.scaler is not None and self.reverse_eval:
            # reshape from (batch_size, seq_len, num_features) to (batch_size * seq_len, num_features)
            outputs = self.datamodule.scaler.inverse_transform(
                outputs.reshape(-1, outputs.shape[-1])
            ).reshape(outputs.shape)
            targets = self.datamodule.scaler.inverse_transform(
                targets.reshape(-1, targets.shape[-1])
            ).reshape(targets.shape)

        # For multivariable forecasting, test should be done on the last variable
        if targets.shape[-1] > 1:
            targets = targets[:, :, [-1]]
            outputs = outputs[:, :, [-1]]

        loss = mse(outputs, targets)
        tar_loss = mse(outputs[:, -1, :], targets[:, -1, :])

        # Append the current results to self.test_step_outputs
        if not hasattr(self, "test_step_outputs"):
            self.test_step_outputs = []
        if self.stop_on_sequence_error:
            self.test_step_outputs.append(loss)
        else:
            self.test_step_outputs.append(tar_loss)

        if not hasattr(self, "test_results"):
            self.test_results = []
        self.test_results.append((targets, outputs))

    def on_test_epoch_end(self):
        metrics = {
            "mae": mase,
            "mse": mse,
            "mape": mape,
            "smape": smape,
        }
        eval_method = ["seq", "tar"]

        # Concatenate the targets and outputs from all test steps
        targets = np.concatenate([o[0] for o in self.test_results])
        outputs = np.concatenate([o[1] for o in self.test_results])
        tar_corr = acc(outputs[:, -1, :], targets[:, -1, :])

        # Calculate the average metrics
        avg_metrics = {}
        for method in eval_method:
            for metric_name, metric_func in metrics.items():
                key = f"{method}_{metric_name}"
                if method == "seq":
                    avg_metrics[key] = (
                        metric_func(outputs, targets, 1.0)
                        if metric_name == "mae"
                        else metric_func(outputs, targets)
                    )
                else:
                    avg_metrics[key] = (
                        metric_func(outputs[:, -1, :], targets[:, -1, :], 1.0)
                        if metric_name == "mae"
                        else metric_func(outputs[:, -1, :], targets[:, -1, :])
                    )

        data_name = self.datamodule.dataset_name.split("_")[0]
        data_title = dataset_title_map[data_name]
        if data_name in ["electricity", "eng", "us2", "us9"]:
            if data_name == "electricity":
                test_season = self.datamodule.dataset_name.split("_")[-1]
                data_title = f"{data_title} - year {test_season}"
            else:
                test_season = self.datamodule.dataset_season
                data_title = f"{data_title} - season {test_season}"

        # Prepare the values dictionary
        values = {
            "seq_mae": avg_metrics["seq_mae"],
            "seq_mse": avg_metrics["seq_mse"],
            "seq_mape": avg_metrics["seq_mape"],
            "seq_smape": avg_metrics["seq_smape"],
            "tar_mae": avg_metrics["tar_mae"],
            "tar_mse": avg_metrics["tar_mse"],
            "tar_mape": avg_metrics["tar_mape"],
            "tar_smape": avg_metrics["tar_smape"],
            "tar_corr": tar_corr,
        }

        # Add a single row with aggregated metrics
        self.log_dict(values)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        if self.lr_scheduler == "cosine_lr":
            scheduler = CosineAnnealingLR(
                optimizer, **self.cosine_annealing_lr_args, verbose=True
            )
        elif self.lr_scheduler == "linear_lr":
            scheduler = LinearLR(optimizer, verbose=True)
        else:
            scheduler = ConstantLR(optimizer, verbose=True)

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
