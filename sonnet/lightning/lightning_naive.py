import numpy as np
import pandas as pd
from typing import List, Tuple

import torch
import lightning as L
from sonnet.utils.metrics import acc
from gluonts.evaluation.metrics import (
    mase,
    mape,
    mse,
    smape,
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
    capitalized_city = ' '.join(word.capitalize() for word in city.replace('_', ' ').split())
    dataset_title_map[city] = f"Forecasting results on temperature 500 hPa in {capitalized_city}"

class BasePersistenceModel(L.LightningModule):
    """
    A persistence model uses historical observation as predictions.
    """

    def __init__(self, configs, datamodule, reverse_eval, **kwargs):
        super().__init__()
        self.datamodule = datamodule
        self.reverse_eval = reverse_eval
        self.predictions: List[torch.Tensor] = []
        self.targets: List[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        """Training is not needed for persistence model"""
        return None

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        """Validation step"""
        x, y = batch
        y_hat = self(x)
        loss = torch.mean((y_hat - y) ** 2)  # MSE
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        """Test step"""
        x, y = batch
        y_hat = self(x)

        # detach
        y = y.detach().cpu().numpy()
        y_hat = y_hat.detach().cpu().numpy()

        # Inverse transform the outputs and targets
        if self.datamodule.scaler is not None and self.reverse_eval:
            # reshape from (batch_size, seq_len, num_features) to (batch_size * seq_len, num_features)
            y = self.datamodule.scaler.inverse_transform(
                y.reshape(-1, y.shape[-1])
            ).reshape(y.shape)
            y_hat = self.datamodule.scaler.inverse_transform(
                y_hat.reshape(-1, y_hat.shape[-1])
            ).reshape(y_hat.shape)

        # Store predictions and targets for later evaluation
        self.predictions.append(y_hat)
        self.targets.append(y)

    def on_test_epoch_end(self):
        """Calculate and log metrics at the end of test epoch"""
        metrics = {
            "mae": mase,
            "mse": mse,
            "mape": mape,
            "smape": smape,
        }
        eval_method = ["seq", "tar"]

        outputs = np.concatenate(self.predictions, axis=0)
        targets = np.concatenate(self.targets, axis=0)
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

        test_time = pd.to_datetime(self.datamodule.dates_iso)
        data_name = self.datamodule.dataset_name.split("_")[0]
        data_target = dataset_target_map[data_name]
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
        table.add_data(*[values[col] for col in columns])
        if self.logger is not None:
            self.logger.experiment.log({"test_metrics_table": table})
        self.log_dict(values)

    def configure_optimizers(self):
        """No optimization needed for persistence model"""
        return None
