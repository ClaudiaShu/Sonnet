import pandas as pd
from typing import Tuple
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L


def get_weather_paths(city: str):
    """Return the path to the weather data for a specific city"""
    path_map = {
        "london": "datasets/weatherbench/london_weather_with_adjacent_1979_2024_6h.csv",
        "newyork": "datasets/weatherbench/newyork_weather_with_adjacent_1979_2024_6h.csv",
        "hongkong": "datasets/weatherbench/hongkong_weather_with_adjacent_1979_2024_6h.csv",
        "capetown": "datasets/weatherbench/capetown_weather_with_adjacent_1979_2024_6h.csv",
        "singapore": "datasets/weatherbench/singapore_weather_with_adjacent_1979_2024_6h.csv",
    }
    return path_map.get(city.lower(), None)


class CustomWeatherDataset(Dataset):
    """
    Dataset for weather forecasting in different cities
    """

    def __init__(
        self,
        path: str,
        seq_length: int = 28,
        pred_length: int = 4,
        mode: str = "train",
        task: str = "MB",  # MT: Multi-variate (all features), MB: Multi-variable (target only)
        city: str = "london",
        end_year: int = 2022,
        start_year: int = 1980,
        scale: bool = True,
        target_column: str = "t850",
        **params,
    ):
        self.seq_length = seq_length
        self.pred_length = pred_length
        self.city = city.lower()
        self.target_column = target_column

        # Define date ranges for train, validation, and test sets
        self.train_start_date = f"{start_year}-01-01"
        self.train_end_date = f"{end_year - 2}-12-31"
        self.val_start_date = f"{end_year - 1}-01-01"
        self.val_end_date = f"{end_year - 1}-12-31"
        self.test_start_date = f"{end_year}-01-01"
        self.test_end_date = f"{end_year}-12-31"

        # Load data
        assert city.lower() in path
        data = pd.read_csv(path, index_col=0)
        data.index = pd.to_datetime(data.index)

        # Make sure target column is the last column for consistency with ILIDataloader
        if (
            self.target_column in data.columns
            and data.columns[-1] != self.target_column
        ):
            print(f"Experiment with target column: {self.target_column}")
            cols = [col for col in data.columns if col != self.target_column] + [
                self.target_column
            ]
            data = data[cols]
        else:
            print(
                f"[IMPORTANT] Experiment with last column: {data.columns[-1]}, as target."
            )
            self.target_column = data.columns[-1]

        # Split data into train, validation, and test sets
        self.train_data, self.val_data, self.test_data = self._data_split(
            data.copy(), y=False, task=task
        )
        self.train_y, self.val_y, self.test_y = self._data_split(
            data.copy(), y=True, task=task
        )

        # Store test time for plotting
        test_time = self.test_y.index[self.pred_length - 1 :]
        self.dates_iso = test_time.strftime("%Y-%m-%dT%H:%M:%S").tolist()

        # Verify dimensions
        assert (
            self.train_data.shape[0] - self.seq_length
            == self.train_y.shape[0] - self.pred_length
        )
        assert (
            self.val_data.shape[0] - self.seq_length
            == self.val_y.shape[0] - self.pred_length
        )
        assert (
            self.test_data.shape[0] - self.seq_length
            == self.test_y.shape[0] - self.pred_length
        )

        # Scale data if needed
        self.scaler = None
        if scale:
            (self.train_data, self.val_data, self.test_data), _ = self._scaling(
                self.train_data, self.val_data, self.test_data
            )
            (self.train_y, self.val_y, self.test_y), scaler_y = self._scaling(
                self.train_y, self.val_y, self.test_y
            )
            self.scaler = scaler_y

        # Select appropriate data based on mode
        if mode == "train":
            self.data = self.train_data.values
            self.y = self.train_y.values
            self.len = self.data.shape[0] - self.seq_length + 1
        elif mode == "val":
            self.data = self.val_data.values
            self.y = self.val_y.values
            self.len = self.data.shape[0] - self.seq_length + 1
        elif mode == "test":
            self.data = self.test_data.values
            self.y = self.test_y.values
            self.len = self.data.shape[0] - self.seq_length + 1

        self.num_variables = self.data.shape[1]

    def _scaling(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        test_data: pd.DataFrame,
    ) -> Tuple[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], StandardScaler]:
        """Scale the data using StandardScaler"""
        scaler = StandardScaler()
        scaler.fit(train_data)
        scaled_train = scaler.transform(train_data)
        scaled_val = scaler.transform(val_data)
        scaled_test = scaler.transform(test_data)
        return (
            pd.DataFrame(
                scaled_train, index=train_data.index, columns=train_data.columns
            ),
            pd.DataFrame(scaled_val, index=val_data.index, columns=val_data.columns),
            pd.DataFrame(scaled_test, index=test_data.index, columns=test_data.columns),
        ), scaler

    def _data_split(
        self,
        data: pd.DataFrame,
        y: bool = False,
        task: str = "MT",
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split the data according to time ranges.
        If y is False: return data for inputs
        If y is True: return data for targets
        """
        h_one = pd.Timedelta(hours=6)  # One time step is 6 hours
        h_seq = pd.Timedelta(
            hours=6 * self.seq_length
        )  # Input sequence length in hours
        h_pred = pd.Timedelta(hours=6 * self.pred_length)  # Prediction length in hours

        train_start = pd.to_datetime(self.train_start_date)
        train_end = pd.to_datetime(self.train_end_date)
        val_start = pd.to_datetime(self.val_start_date)
        val_end = pd.to_datetime(self.val_end_date)
        test_start = pd.to_datetime(self.test_start_date)
        test_end = pd.to_datetime(self.test_end_date)

        if not y:
            # For inputs
            train_data = data.loc[
                train_start - h_seq - h_pred + h_one : train_end - h_pred
            ].copy()
            val_data = data.loc[
                val_start - h_seq - h_pred + h_one : val_end - h_pred
            ].copy()
            test_data = data.loc[
                test_start - h_seq - h_pred + h_one : test_end - h_pred
            ].copy()
        else:
            # For targets
            if task == "MT":
                # Targets are all features
                data_y = data.copy()
            else:
                # Target is only the target column (e.g., temperature)
                data_y = data[self.target_column].copy().to_frame()

            train_data = data_y.loc[train_start - h_pred + h_one : train_end].copy()
            val_data = data_y.loc[val_start - h_pred + h_one : val_end].copy()
            test_data = data_y.loc[test_start - h_pred + h_one : test_end].copy()

        return train_data, val_data, test_data

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return (
            torch.tensor(self.data[index : index + self.seq_length], dtype=torch.float),
            torch.tensor(self.y[index : index + self.pred_length], dtype=torch.float),
        )


class CustomWeatherDataModule(L.LightningDataModule):
    def __init__(
        self,
        city_year: str,
        batch_size: int = 32,
        **params,
    ):
        super().__init__()
        parts = city_year.lower().split("_")
        if len(parts) != 2:
            raise ValueError(
                f"City_year {city_year} is not in the expected format 'city_year'."
            )

        city, year_str = parts
        valid_cities = {"london", "newyork", "hongkong", "capetown", "singapore"}
        if city not in valid_cities:
            raise ValueError(
                f"City {city} not recognized. Valid cities: {', '.join(valid_cities)}."
            )
        self.dataset_name = city_year

        try:
            self.end_year = int(year_str)
        except ValueError:
            raise ValueError(
                f"Year part '{year_str}' in city_year {city_year} is not numeric."
            )

        self.batch_size = batch_size
        self.params = params
        self.params["city"] = city
        self.params["end_year"] = self.end_year

    def setup(self, stage=None):
        """
        Stage: fit, test, predict
        """
        path = get_weather_paths(self.params["city"])
        if path is None:
            raise ValueError(f"Path not found for city: {self.params['city']}")

        self.data_train = CustomWeatherDataset(path, **self.params, mode="train")
        self.data_val = CustomWeatherDataset(path, **self.params, mode="val")
        self.data_test = CustomWeatherDataset(path, **self.params, mode="test")
        self.scaler = self.data_train.scaler
        self.num_variables = self.data_train.num_variables
        self.dates_iso = self.data_test.dates_iso

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size, shuffle=False)


if __name__ == "__main__":
    params = {
        "seq_length": 28,
        "pred_length": 4,
        "scale": True,
        "task": "MT",
        "target_column": "t850",
        "start_year": 1980,
    }

    weather_data_module = CustomWeatherDataModule(
        "london_2018", batch_size=64, **params
    )
    weather_data_module.setup()

    train_loader = weather_data_module.train_dataloader()
    val_loader = weather_data_module.val_dataloader()
    test_loader = weather_data_module.test_dataloader()

    # Print sample batch shapes
    first_batch = next(iter(train_loader))
    print("Train batch shapes:")
    print(f"  Inputs: {first_batch[0].shape}")
    print(f"  Targets: {first_batch[1].shape}")

    first_batch = next(iter(val_loader))
    print("Validation batch shapes:")
    print(f"  Inputs: {first_batch[0].shape}")
    print(f"  Targets: {first_batch[1].shape}")

    first_batch = next(iter(test_loader))
    print("Test batch shapes:")
    print(f"  Inputs: {first_batch[0].shape}")
    print(f"  Targets: {first_batch[1].shape}")

    # Print dataset sizes
    print(f"Train dataset size: {len(weather_data_module.data_train)}")
    print(f"Validation dataset size: {len(weather_data_module.data_val)}")
    print(f"Test dataset size: {len(weather_data_module.data_test)}")

    # Print data ranges
    print(
        f"Train data range: {weather_data_module.data_train.train_start_date} to {weather_data_module.data_train.train_end_date}"
    )
    print(
        f"Val data range: {weather_data_module.data_val.val_start_date} to {weather_data_module.data_val.val_end_date}"
    )
    print(
        f"Test data range: {weather_data_module.data_test.test_start_date} to {weather_data_module.data_test.test_end_date}"
    )
