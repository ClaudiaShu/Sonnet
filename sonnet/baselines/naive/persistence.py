import numpy as np
import pandas as pd
from typing import List, Optional, Union, Tuple
import torch
import lightning as L
from sonnet.utils.metrics import acc
from gluonts.evaluation.metrics import (
    mase,
    mape,
    mse,
    smape,
)

from sonnet.lightning.lightning_naive import BasePersistenceModel


class Model(BasePersistenceModel):
    """
    A seasonal persistence model that uses historical seasonal patterns to make predictions.
    The model assumes that the pattern from the previous season will repeat.
    """

    def __init__(self, configs, **kwargs):
        super(Model, self).__init__(configs, **kwargs)
        self.seq_length = configs.seq_len
        self.pred_length = configs.pred_len
        self.target_idx = -1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Makes predictions using simple persistence.
        The model assumes the last observed value will persist into the future.

        Args:
            x: Input tensor of shape (batch_size, seq_length, num_features)

        Returns:
            Predictions tensor of shape (batch_size, pred_length, 1)
        """
        # Extract target variable
        x_target = x[:, :, self.target_idx]

        # Get the last value for each batch
        last_values = x_target[:, -1].unsqueeze(1)  # Shape: (batch_size, 1)

        # Repeat the last value pred_length times
        predictions = last_values.repeat(
            1, self.pred_length
        )  # Shape: (batch_size, pred_length)

        return predictions.unsqueeze(
            -1
        )  # Add feature dimension to match expected shape


class PersistenceModel(L.LightningModule):
    """
    A seasonal persistence model that uses historical seasonal patterns to make predictions.
    The model assumes that the pattern from the previous season will repeat.
    """

    def __init__(
        self,
        seq_length: int,
        pred_length: int,
        seasonality: int = None,  # If None, will use seq_length as seasonality
        target_idx: int = -1,  # Index of target variable (last column by default)
    ):
        super().__init__()
        self.seq_length = seq_length
        self.pred_length = pred_length
        self.seasonality = seasonality if seasonality else seq_length
        self.target_idx = target_idx
        self.predictions: List[torch.Tensor] = []
        self.targets: List[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Makes predictions using seasonal persistence.

        Args:
            x: Input tensor of shape (batch_size, seq_length, num_features)

        Returns:
            Predictions tensor of shape (batch_size, pred_length, 1)
        """
        batch_size = x.shape[0]

        # Extract target variable
        x_target = x[:, :, self.target_idx]

        # Calculate predictions using seasonal persistence
        predictions = []
        for i in range(self.pred_length):
            # Use values from one season ago
            idx = -(self.seasonality - i)
            if abs(idx) > x_target.shape[1]:
                # If we don't have enough history, use the last available value
                pred = x_target[:, -1].unsqueeze(1)
            else:
                pred = x_target[:, idx].unsqueeze(1)
            predictions.append(pred)

        # Stack predictions along time dimension
        predictions = torch.cat(predictions, dim=1)

        return predictions.unsqueeze(-1)  # Add feature dimension

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

        # Store predictions and targets for later evaluation
        self.predictions.append(y_hat.detach().cpu())
        self.targets.append(y.detach().cpu())

        loss = torch.mean((y_hat - y) ** 2)  # MSE
        return loss

    def on_test_epoch_end(self):
        """Calculate and log metrics at the end of test epoch"""
        metrics = {
            "mae": mase,
            "mse": mse,
            "mape": mape,
            "smape": smape,
        }
        eval_method = ["seq", "tar"]

        outputs = torch.cat(self.predictions, dim=0).numpy()
        targets = torch.cat(self.targets, dim=0).numpy()
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

        self.log_dict(avg_metrics)
        self.log("tar_corr", tar_corr)

    def configure_optimizers(self):
        """No optimization needed for persistence model"""
        return None


def run_persistence(
    datamodule: L.LightningDataModule,
    seq_length: int,
    pred_length: int,
    seasonality: Optional[int] = None,
) -> dict:
    """
    Run seasonal persistence model on given datamodule

    Args:
        datamodule: Lightning datamodule containing the dataset
        seq_length: Length of input sequence
        pred_length: Length of prediction horizon
        seasonality: Seasonal period (defaults to seq_length if None)

    Returns:
        Dictionary containing test metrics
    """

    # Initialize model
    model = PersistenceModel(seq_length=seq_length, pred_length=pred_length)

    # Initialize trainer
    trainer = L.Trainer(
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        devices=[0],
    )

    # Test model
    results = trainer.test(model, datamodule=datamodule)

    return results[0]


if __name__ == "__main__":
    from sonnet.data.dartsDataloader import CustomDartsDataModule

    # Example usage with ETTh1 dataset
    params = {
        "seq_length": 96,
        "pred_length": 24,
        "do_dataset_split_ratio": False,
        "dataset_split_numeric": (500, 100, 100),
        "scale": True,
    }

    # Initialize datamodule
    datamodule = CustomDartsDataModule("etth1", batch_size=32, **params)

    # Run model
    results = run_persistence(
        datamodule=datamodule,
        seq_length=params["seq_length"],
        pred_length=params["pred_length"],
    )
