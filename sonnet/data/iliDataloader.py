import pandas as pd
import datetime as dt
from typing import Tuple
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L

from sonnet.data.dataset_utils_ili import (
    obtain_peak,
    obtain_onset,
    obtain_drop,
    baseline,
    corr_with,
    get_ili_paths,
)


class CustomILIDataset(Dataset):
    """
    eng:
        delay: 7 days
        start: "09-01"
        end: "08-31"
    us:
        delay: 14 days
        start: "08-01"
        end: "07-31"
    """

    def __init__(
        self,
        path: str,
        seq_length: int = 28,
        pred_length: int = 7,
        top_c: int = 64,
        mode: str = "train",
        task: str = "MT",  # MT: Multi-variate, MB: Multi-variable
        region: str = "eng",  # eng or us2, us9
        end_year: int = 2016,
        total_year: int = 11,
        corr_year: int = 5,
        scale: bool = True,
        **params,
    ):
        self.seq_length = seq_length
        self.pred_length = pred_length

        self.region = region

        self.rate_lag = 7 if region == "eng" else 14
        self.train_start_date = (
            f"{end_year - total_year}-09-01"
            if region == "eng"
            else f"{end_year - total_year}-08-01"
        )
        self.train_end_date = (
            f"{end_year - 1}-08-31" if region == "eng" else f"{end_year - 1}-07-31"
        )
        self.test_start_date = (
            f"{end_year - 1}-09-01" if region == "eng" else f"{end_year - 1}-08-01"
        )
        self.test_end_date = (
            f"{end_year}-08-31" if region == "eng" else f"{end_year}-07-31"
        )
        self.corr_start_date = (
            f"{end_year - 1 - corr_year}-09-01"
            if region == "eng"
            else f"{end_year - 1 - corr_year}-08-01"
        )

        # onset
        self.season_start_date1 = (
            f"{end_year - 2}-09-01" if region == "eng" else f"{end_year - 2}-08-01"
        )
        self.season_end_date1 = (
            f"{end_year - 1}-08-31" if region == "eng" else f"{end_year - 1}-07-31"
        )
        # peak
        self.season_start_date2 = (
            f"{end_year - 3}-09-01" if region == "eng" else f"{end_year - 3}-08-01"
        )
        self.season_end_date2 = (
            f"{end_year - 2}-08-31" if region == "eng" else f"{end_year - 2}-07-31"
        )
        # drop
        self.season_start_date3 = (
            f"{end_year - 4}-09-01" if region == "eng" else f"{end_year - 4}-08-01"
        )
        self.season_end_date3 = (
            f"{end_year - 3}-08-31" if region == "eng" else f"{end_year - 3}-07-31"
        )

        # Obtain the data
        assert region in path
        data = pd.read_csv(path, index_col=0)
        data.index = pd.to_datetime(data.index)

        # filter columns based on the correlation
        data = self._corr_filter(data, top_c=top_c)

        # Split the data into train, validation, and test sets
        val_start_tuple = self._get_val_indices(region, data["rate"].copy())
        self.train_data, self.val_data, self.test_data = self._data_split(
            data.copy(), val_start_tuple
        )
        self.train_y, self.val_y, self.test_y = self._data_split(
            data.copy(), val_start_tuple, y=True, task=task
        )
        test_time = self.test_y.index[self.pred_length - 1 :]
        self.dates_iso = test_time.strftime("%Y-%m-%dT%H:%M:%S").tolist()

        assert (
            self.train_data.shape[0] - self.seq_length
            == self.train_y.shape[0] - self.pred_length
        )
        assert (
            self.val_data.shape[0] - self.seq_length * 3
            == self.val_y.shape[0] - self.pred_length * 3
        )
        assert (
            self.test_data.shape[0] - self.seq_length
            == self.test_y.shape[0] - self.pred_length
        )

        self.scaler = None
        if scale:
            (self.train_data, self.val_data, self.test_data), _ = self._scaling(
                self.train_data, self.val_data, self.test_data
            )
            (self.train_y, self.val_y, self.test_y), scaler_y = self._scaling(
                self.train_y, self.val_y, self.test_y
            )
            self.scaler = scaler_y

        if mode == "train":
            self.data = self.train_data.values
            self.y = self.train_y.values
            self.len = self.data.shape[0] - self.seq_length + 1
            self.slice = 0
        elif mode == "val":
            self.data = self.val_data.values
            self.y = self.val_y.values
            self.len = self.data.shape[0] - self.seq_length * 3 + 3
            self.slice = 3
        elif mode == "test":
            self.data = self.test_data.values
            self.y = self.test_y.values
            self.len = self.data.shape[0] - self.seq_length + 1
            self.slice = 0

        self.num_variables = self.data.shape[1]

    def _corr_filter(self, data: pd.DataFrame, top_c=64):
        """
        Filter columns based on correlation with 'rate'.

        Args:
            data: DataFrame containing the data
            top_c: Either an integer to select top N features, or a float threshold
                for correlation filtering (features with correlation > top_c are selected)

        Returns:
            DataFrame with filtered columns
        """
        # Correlate with the previous years
        data_corr = data.loc[self.corr_start_date : self.train_end_date].copy()
        corr_filtered, sorted_correlation = corr_with(data_corr, "rate")

        if isinstance(top_c, int):
            # Select top N features by correlation (as before)
            # C + 1 to include the target column
            keys = sorted_correlation.index[: top_c + 1]
            if "rate" in keys:
                keys = keys.drop("rate").append(pd.Index(["rate"]))
            print(f"Top {top_c} features selected based on correlation.")
        elif isinstance(top_c, float) and 0 < top_c < 1:
            # top_c is a float threshold
            # Select features with correlation > threshold
            keys = sorted_correlation[abs(sorted_correlation) > top_c].index
            if "rate" not in keys:
                keys = keys.append(pd.Index(["rate"]))
            else:
                keys = keys.drop("rate").append(pd.Index(["rate"]))
            print(
                f"{len(keys)} features selected based on correlation threshold: {top_c}."
            )
        else:
            raise ValueError(
                "top_c must be either an integer or a float between 0 and 1."
            )

        return data[keys]

    def _scaling(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        test_data: pd.DataFrame,
    ) -> Tuple[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], StandardScaler]:
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
        val_start_tuple: Tuple,
        y: bool = False,
        task: str = "MT",
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split the data according to time ranges.
        If y is False:
        - train data is selected between self.start_date and self.train_end_date.
        - test data is selected between self.test_start_date and self.test_end_date.
        Additionally, for the train data, delay the "rate" column by lag time steps,
        filling the resulting NaN values with zero.
        """
        onset_start, peak_start, drop_start = val_start_tuple
        d_one = pd.Timedelta(days=1)
        d_seq = pd.Timedelta(days=self.seq_length)
        d_pred = pd.Timedelta(days=self.pred_length)
        val_period = pd.Timedelta(days=59)  # 60 - 1 as the last day is included

        train_start = pd.to_datetime(self.train_start_date)
        train_end = pd.to_datetime(self.train_end_date)
        test_start = pd.to_datetime(self.test_start_date)
        test_end = pd.to_datetime(self.test_end_date)

        if not y:
            # Shift the "rate" column by lag of time steps and fill NaN values with 0
            data["rate"] = data["rate"].shift(self.rate_lag).fillna(0)
            # data = data.drop(columns=["rate"])

            train_data = data.loc[
                train_start - d_seq - d_pred + d_one : train_end - d_pred
            ].copy()
            test_data = data.loc[
                test_start - d_seq - d_pred + d_one : test_end - d_pred
            ].copy()

            val_data_onset = data.loc[
                onset_start - d_seq - d_pred + d_one : onset_start + val_period - d_pred
            ]
            val_data_peak = data.loc[
                peak_start - d_seq - d_pred + d_one : peak_start + val_period - d_pred
            ]
            val_data_drop = data.loc[
                drop_start - d_seq - d_pred + d_one : drop_start + val_period - d_pred
            ]

        else:
            if task == "MT":
                # Targets are rate and google search queries without lag
                data_y = data.copy()
            else:
                # Targets are ili rate
                data_y = data["rate"].copy().to_frame()
            train_data = data_y.loc[train_start - d_pred + d_one : train_end].copy()
            test_data = data_y.loc[test_start - d_pred + d_one : test_end].copy()

            val_data_onset = data_y.loc[
                onset_start - d_pred + d_one : onset_start + val_period
            ]
            val_data_peak = data_y.loc[
                peak_start - d_pred + d_one : peak_start + val_period
            ]
            val_data_drop = data_y.loc[
                drop_start - d_pred + d_one : drop_start + val_period
            ]

        val_data = pd.concat([val_data_onset, val_data_peak, val_data_drop])
        rm_onset = data.loc[onset_start : onset_start + val_period]
        rm_peak = data.loc[peak_start : peak_start + val_period]
        rm_drop = data.loc[drop_start : drop_start + val_period]
        rm_data = pd.concat([rm_onset, rm_peak, rm_drop])

        train_data = train_data.drop(rm_data.index)

        return train_data, val_data, test_data

    def _get_range(self, data: pd.DataFrame, start_date: str, end_date: str):
        return data.loc[start_date:end_date].copy()

    def _get_before(self, data: pd.DataFrame, date: str):
        return data.loc[:date].copy()

    def _get_val_indices(self, region: str, data_y: pd.DataFrame):
        season_data1 = self._get_range(
            data_y, self.season_start_date1, self.season_end_date1
        )
        season_data2 = self._get_range(
            data_y, self.season_start_date2, self.season_end_date2
        )
        season_data3 = self._get_range(
            data_y, self.season_start_date3, self.season_end_date3
        )

        baseline1 = self._calculate_baseline(
            region, data_y, self.season_start_date1, self.season_end_date1
        )
        baseline3 = self._calculate_baseline(
            region, data_y, self.season_start_date3, self.season_end_date3
        )

        onset_centre = obtain_onset(season_data1, window=14, threshold=baseline1)
        peak_centre = obtain_peak(season_data2)
        drop_centre = obtain_drop(season_data3, window=14, threshold=baseline3)

        # days_before = len(self.get_before(data_rate, self.start_date))
        # offset = pd.Timedelta(days=(30 + (self.seq_length + 1) + self.pred_length))
        offset = pd.Timedelta(days=30)
        onset_start = onset_centre - offset
        peak_start = peak_centre - offset
        drop_start = drop_centre - offset

        return (
            onset_start,
            peak_start,
            drop_start,
        )

    def _get_non_influenza_weeks(
        self, region, data_whole, season_start_date, season_end_date
    ):
        one_y = dt.timedelta(days=365)
        start_dt = pd.to_datetime(season_start_date)
        end_dt = pd.to_datetime(season_end_date)
        season_data = self._get_range(
            data_whole, start_dt - one_y - one_y, end_dt - one_y
        )
        # Filter weeks with less than 2% positive specimens
        non_flu = 6 if region == "eng" else 2
        # non_flu = 3.5 if region == "eng" else 1.5
        non_influenza_weeks = season_data[season_data < non_flu]
        return non_influenza_weeks

    def _calculate_baseline(
        self, region, data_whole, season_start_date, season_end_date
    ):
        non_influenza_weeks = self._get_non_influenza_weeks(
            region, data_whole, season_start_date, season_end_date
        )
        return baseline(non_influenza_weeks)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        if self.slice == 0:
            return (
                torch.tensor(
                    self.data[index : index + self.seq_length], dtype=torch.float
                ),
                torch.tensor(
                    self.y[index : index + self.pred_length], dtype=torch.float
                ),
            )
        elif self.slice == 3:
            if index < 60:
                data_slice = self.data[index : index + self.seq_length]
                y_slice = self.y[index : index + self.pred_length]
            elif 60 <= index < 120:
                data_slice = self.data[
                    index + self.seq_length - 1 : index + self.seq_length * 2 - 1
                ]
                y_slice = self.y[
                    index + self.pred_length - 1 : index + self.pred_length * 2 - 1
                ]
            else:
                data_slice = self.data[
                    index + self.seq_length * 2 - 2 : index + self.seq_length * 3 - 2
                ]
                y_slice = self.y[
                    index + self.pred_length * 2 - 2 : index + self.pred_length * 3 - 2
                ]

            return (
                torch.tensor(data_slice, dtype=torch.float),
                torch.tensor(y_slice, dtype=torch.float),
            )


class CustomILIDataModule(L.LightningDataModule):
    def __init__(
        self,
        region: str,
        batch_size: int = 12,
        **params,
    ):
        super().__init__()
        parts = region.lower().split("_")
        if len(parts) != 2:
            raise ValueError(
                f"Region {region} is not in the expected format 'code_year'."
            )

        region_code, season_str = parts
        valid_regions = {"eng", "us2", "us9"}
        if region_code not in valid_regions:
            raise ValueError(
                f"Region {region} not recognized. Valid regions: {', '.join(valid_regions)}."
            )
        self.dataset_name = region_code

        try:
            season = int(season_str)
        except ValueError:
            raise ValueError(
                f"Season part '{season_str}' in region {region} is not numeric."
            )
        self.dataset_season = f"{season - 1} / {season}"

        self.batch_size = batch_size
        self.params = params

    def setup(self, stage=None):
        """
        Stage: fit, test, predict
        """
        path = get_ili_paths(self.dataset_name)
        self.data_train = CustomILIDataset(
            path, region=self.dataset_name, **self.params, mode="train"
        )
        self.data_val = CustomILIDataset(
            path, region=self.dataset_name, **self.params, mode="val"
        )
        self.data_test = CustomILIDataset(
            path, region=self.dataset_name, **self.params, mode="test"
        )
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
        "seq_length": 56,
        "pred_length": 28,
        "scale": True,
        "region": "us2_2017",
        "end_year": 2017,
        "total_year": 10,
        "corr_year": 5,
    }

    ili_data_module = CustomILIDataModule(batch_size=64, **params)
    ili_data_module.setup()
    train_loader = ili_data_module.train_dataloader()
    val_loader = ili_data_module.val_dataloader()

    test_loader = ili_data_module.test_dataloader()
    first_batch = next(iter(train_loader))
    print(first_batch[0].shape)
    print(first_batch[1].shape)
    first_batch = next(iter(val_loader))
    print(first_batch[0].shape)
    first_batch = next(iter(test_loader))
    print(first_batch[0].shape)

    # Iterate through train_loader
    print("Train Loader:")
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        print(f"Batch {batch_idx}:")
        print(f"  Inputs shape: {inputs.shape}")
        print(f"  Targets shape: {targets.shape}")

    # Iterate through val_loader
    print("\nValidation Loader:")
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        print(f"Batch {batch_idx}:")
        print(f"  Inputs shape: {inputs.shape}")
        print(f"  Targets shape: {targets.shape}")

    # Iterate through test_loader
    print("\nTest Loader:")
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        print(f"Batch {batch_idx}:")
        print(f"  Inputs shape: {inputs.shape}")
        print(f"  Targets shape: {targets.shape}")
