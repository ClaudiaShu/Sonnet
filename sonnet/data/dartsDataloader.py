import pandas as pd
from typing import List, Tuple, Union
from sklearn.preprocessing import StandardScaler

from darts import concatenate
from darts import TimeSeries  # type: ignore
from darts.datasets import (  # type: ignore
    AirPassengersDataset,
    AusBeerDataset,
    AustralianTourismDataset,
    EnergyDataset,
    GasRateCO2Dataset,
    HeartRateDataset,
    IceCreamHeaterDataset,
    MonthlyMilkDataset,
    MonthlyMilkIncompleteDataset,
    SunspotsDataset,
    TaylorDataset,
    TemperatureDataset,
    USGasolineDataset,
    WineDataset,
    WoolyDataset,
    ETTh1Dataset,
    ETTh2Dataset,
    ETTm1Dataset,
    ETTm2Dataset,
    TaxiNewYorkDataset,
    ElectricityDataset,
    UberTLCDataset,
    ILINetDataset,
    ExchangeRateDataset,
    TrafficDataset,
    WeatherDataset,
    ElectricityConsumptionZurichDataset,
)
from darts.utils.missing_values import fill_missing_values, missing_values_ratio
from sonnet.data import dataset_list_darts

import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L
import warnings

# todo: scaler should be able to inverse_transform


# Get the dataset by name
def get_darts_series(name: str) -> TimeSeries:
    print(f"Loading {name} dataset")
    if name == "airpassengers":
        return AirPassengersDataset().load()
    elif name == "ausbeer":
        return AusBeerDataset().load()
    elif name == "australian_tourism":
        return AustralianTourismDataset().load()
    elif name == "energy":
        """
        For energy dataset, some variables have all 0 values or NaN values. 
        These columns do not provide any useful information for time series 
        forecasting and thus are removed. 

        By removing these columns, we ensure that the dataset used for time 
        series forecasting is clean and free from irrelevant or misleading 
        information. This preprocessing step is crucial for improving the 
        performance and accuracy of the forecasting model. The remaining 
        columns in the dataset provide meaningful and relevant information 
        that can help the model learn the underlying patterns and make 
        accurate predictions.

        Target: price actual

        Freq: Hourly

        Split:
            train: 2014-12-31 23:00:00 to 2017-05-31 23:00:00   (21169) (17545)
            val: 2017-06-01 00:00:00 to 2017-12-31 23:00:00     (5136)  (8760)
            test: 2018-01-01 00:00:00 to 2018-12-31 22:00:00    (8759)  (8759)
        
        Horizons:
            24/48/72/168 for 1/2/3/7 days

        Description: Contains a time series with 28 hourly components between 
        2014-12-31 23:00:00 and 2018-12-31 22:00:00

        Ref: https://www.kaggle.com/nicholasjhana/energy-consumption-generation-prices-and-weather
        """
        series = EnergyDataset().load()
        series = series.drop_columns(
            [
                "generation fossil coal-derived gas",  # All 0 values
                "generation fossil oil shale",  # All 0 values
                "generation fossil peat",  # All 0 values
                "generation geothermal",  # All 0 values
                "generation hydro pumped storage aggregated",  # Nan values
                "generation marine",  # All 0 values
                "generation wind offshore",  # All 0 values
                "forecast wind offshore eday ahead",  # Nan values
            ]
        )
        return series
    elif name == "gas_rate_co2":
        return GasRateCO2Dataset().load()
    elif name == "heart_rate":
        return HeartRateDataset().load()
    elif name == "ice_cream_heater":
        """
        Target: ice cream production

        Freq: Monthly

        Split:
            train: 2004-01-01 to 2015-12-01 (144 months)
            val: 2016-01-01 to 2017-12-01 (24 months)
            test: 2018-01-01 to 2019-12-01 (24 months)

        horizon:
            1/6/12 months

        Description: Monthly sales of heaters and ice cream between January 
        2004-01-01 and June 2020-06-01.
        """
        return IceCreamHeaterDataset().load()
    elif name == "monthly_milk":
        return MonthlyMilkDataset().load()
    elif name == "monthly_milk_incomplete":
        return MonthlyMilkIncompleteDataset().load()
    elif name == "sunspots":
        return SunspotsDataset().load()
    elif name == "taylor":
        return TaylorDataset().load()
    elif name == "temperature":
        return TemperatureDataset().load()
    elif name == "us_gasoline":
        return USGasolineDataset().load()
    elif name == "wine":
        return WineDataset().load()
    elif name == "wooly":
        return WoolyDataset().load()
    elif name == "etth1":
        return ETTh1Dataset().load()
    elif name == "etth2":
        return ETTh2Dataset().load()
    elif name == "ettm1":
        return ETTm1Dataset().load()
    elif name == "ettm2":
        return ETTm2Dataset().load()
    elif name == "taxi_nyc":
        return TaxiNewYorkDataset().load()
    elif name == "electricity":
        return ElectricityDataset().load()
    elif name == "uber_tlc":
        return UberTLCDataset().load()
    elif name == "ili_net":
        return ILINetDataset().load()
    elif name == "exchange_rate":
        return ExchangeRateDataset().load()
    elif name == "traffic":
        return TrafficDataset().load()
    elif name == "weather":
        return WeatherDataset().load()
    elif name == "electricity_consumption_zurich_2020":
        """
        Components: 
            Value_NE5 : Households & SMEs electricity consumption (low voltage, grid level 7) in kWh
            Value_NE7 : Business and services electricity consumption (medium voltage, grid level 5) in kWh
            Hr [%Hr] : Relative humidity
            RainDur [min] : Duration of precipitation (divided by 4 for conversion from hourly to quarter-hourly records)
            T [°C] : Temperature
            WD [°] : Wind direction
            WVv [m/s] : Wind vector speed
            p [hPa] : Air pressure
            WVs [m/s] : Wind scalar speed
            StrGlo [W/m2] : Global solar irradiation

        Target: 'Value_NE5'
        (https://unit8co.github.io/darts/examples/23-Conformal-Prediction-examples.html?highlight=zurich)

        Freq: 15 minutes

        Description: 
        Electricity Consumption of households & SMEs (low voltage) and 
        businesses & services (medium voltage) in the city of Zurich [1], 
        with values recorded every 15 minutes.

        The electricity consumption is combined with weather measurements 
        recorded by three different stations in the city of Zurich with 
        a hourly frequency [2]. The missing time stamps are filled with 
        NaN. The original weather data is recorded every hour. Before 
        adding the features to the electricity consumption, the data is 
        resampled to 15 minutes frequency, and missing values are interpolated.

        To simplify the dataset, the measurements from the Zch_Schimmelstrasse 
        and Zch_Rosengartenstrasse weather stations are discarded to keep 
        only the data recorded in the Zch_Stampfenbachstrasse station.

        Both dataset sources are updated continuously, but this dataset 
        only retrains values between 2015-01-01 and 2022-08-31. The time 
        index was converted from CET time zone to UTC.

        split: 
            train: 2015-01-01 to 2018-12-31 (1461 days) (140256 if 15 min)  (35064)
            val: 2019-01-01 to 2019-12-31 (365)         (35040)             (8760)
            test: 2020-01-01 to 2020-12-31 (366)        (35136)             (8784)

            train: 2016-01-01 to 2019-12-31 (1461 days) (140256 if 15 min)  (35064)
            val: 2020-01-01 to 2020-12-31 (366)         (35136)             (8784)
            test: 2021-01-01 to 2021-12-31 (366)        (35040)             (8760)

        total length = 24 * 4 * 2799
        Note: before 2018, the scalar speeds were calculated from the 30 minutes vector data.

        horizons: 1/2/3/4 weeks,
        (potentially could consider resample to 1 hour)
        (that'll be 168/336/504/672 time steps)

        References
        https://data.stadt-zuerich.ch/dataset/ewz_stromabgabe_netzebenen_stadt_zuerich
        https://data.stadt-zuerich.ch/dataset/ugz_meteodaten_stundenmittelwerte

        """
        series = ElectricityConsumptionZurichDataset().load().resample(freq="h")
        ne5 = series["Value_NE5"]
        series = series.drop_columns(["Value_NE5"])
        series = concatenate([series, ne5], axis=1)
        return series
    elif name == "electricity_consumption_zurich_2021":
        series = ElectricityConsumptionZurichDataset().load().resample(freq="h")
        ne5 = series["Value_NE5"]
        series = series.drop_columns(["Value_NE5"])
        series = concatenate([series, ne5], axis=1)
        series = series.slice(pd.Timestamp("2016-01-01"), pd.Timestamp("2022-01-01"))
        return series
    else:
        raise ValueError("Invalid dataset name")


# Custom dataset from darts
class CustomDartsDataset(Dataset):
    def __init__(
        self,
        time_series: TimeSeries,
        seq_length: int = 10,
        pred_length: int = 6,
        mode: str = "train",
        task: str = "MT",  # MT: Multi-variate, MB: Multi-variable
        do_dataset_split_ratio: bool = True,
        dataset_split: Union[Tuple[float, float], Tuple[float, float, float]] = (
            0.7,
            0.1,
            0.2,
        ),
        dataset_split_numeric: Union[Tuple[int, int], Tuple[int, int, int]] = (
            100,
            20,
            20,
        ),
        scale: bool = True,
        **params,
    ):
        self.seq_length = seq_length
        self.pred_length = pred_length
        # meta data
        freq = time_series.freq_str
        comp = time_series.components.tolist()
        self.metadata = {"freq": freq, "num_instances": len(comp), "comp": comp}
        print(f"Frequency: {freq}, Components: {len(comp)}, Length: {len(time_series)}")

        # ! for time series with partial missing values
        time_series = fill_missing_values(time_series)
        assert not missing_values_ratio(time_series) > 0, (
            "Missing values in the dataset"
        )

        self.scaler = None
        if scale:
            train_len = int(len(time_series) * dataset_split[0])
            time_series, scaler = self._scaling(time_series, train_len)
            self.scaler = scaler

        if do_dataset_split_ratio:
            warnings.warn(
                "Using dataset split ratio is suboptimal. Consider using numeric split instead.",
                UserWarning,
            )
            self.train_data, self.val_data, self.test_data = self._data_split(
                time_series, split=dataset_split
            )
        else:
            self.train_data, self.val_data, self.test_data = self._data_split_numeric(
                time_series, split=dataset_split_numeric
            )
        test_time = self.test_data.time_index[self.seq_length + self.pred_length :]
        self.dates_iso = test_time.strftime("%Y-%m-%dT%H:%M:%S").tolist()

        if mode == "train":
            self.data = self.train_data.values()
        elif mode == "val":
            self.data = self.val_data.values()
        elif mode == "test":
            self.data = self.test_data.values()
        else:
            raise ValueError("Invalid model argument")

        if task == "MB":
            #  need to specify the target variable
            self.data_y = self.data[:, [-1]]
        else:
            self.data_y = self.data

    def _scaling(
        self, series: TimeSeries, train_len: int
    ) -> Tuple[TimeSeries, StandardScaler]:
        t_data = series.slice_intersect(series[:train_len])
        scaler = StandardScaler()
        # Fit the scaler on the training data
        scaler.fit(t_data.values())
        # Scale the entire series
        scaled_series = scaler.transform(series.values())
        if series.has_datetime_index:
            return TimeSeries.from_times_and_values(
                series.time_index, scaled_series
            ), scaler
        else:
            return TimeSeries.from_values(scaled_series), scaler

    def _data_split(
        self,
        data: TimeSeries,
        split: List[float] = [0.7, 0.1, 0.2],
    ) -> Tuple[TimeSeries, ...]:
        """
        Split a TimeSeries object into subset TimeSeries objects
        :param data: The TimeSeries object to split
        :param split: The fraction of the data to be used for splitting
        :return: A tuple of TimeSeries objects

        Options:
        - len(split) == 2: return train_data, test_data
        - len(split) == 3: return train_data, val_data, test_data
        """
        assert sum(split) == 1
        assert len(split) in [2, 3]
        train_data = data.slice_intersect(data[: int(len(data) * split[0])])

        if len(split) == 2:
            test_data = data.slice_intersect(data[int(len(data) * split[0]) :])
            return train_data, test_data
        elif len(split) == 3:
            val_data = data.slice_intersect(
                data[int(len(data) * split[0]) : int(len(data) * (split[0] + split[1]))]
            )
            test_data = data.slice_intersect(
                data[int(len(data) * (split[0] + split[1])) :]
            )
            return train_data, val_data, test_data
        else:
            raise ValueError("Invalid split argument")

    def _data_split_numeric(
        self,
        data: TimeSeries,
        split: List[int] = [100, 20, 20],
    ) -> Tuple[TimeSeries, ...]:
        """
        Split a TimeSeries object into subset TimeSeries objects
        :param data: The TimeSeries object to split
        :param split: The number of data points to be used for splitting
        :return: A tuple of TimeSeries objects

        Options:
        - len(split) == 2: return train_data, test_data
        - len(split) == 3: return train_data, val_data, test_data
        """
        assert sum(split) <= len(data)
        assert len(split) in [2, 3]
        train_data = data.slice_intersect(data[: split[0]])

        if len(split) == 2:
            test_data = data.slice_intersect(data[split[0] :])
            return train_data, test_data
        elif len(split) == 3:
            val_data = data.slice_intersect(
                data[
                    split[0] - self.seq_length - self.pred_length : split[0] + split[1]
                ]
            )
            test_data = data.slice_intersect(
                data[
                    split[0] + split[1] - self.seq_length - self.pred_length : split[0]
                    + split[1]
                    + split[2]
                ]
            )
            return train_data, val_data, test_data
        else:
            raise ValueError("Invalid split argument")

    def __len__(self):
        return len(self.data) - self.seq_length - self.pred_length

    def __getitem__(self, index):
        return (
            torch.tensor(self.data[index : index + self.seq_length], dtype=torch.float),
            torch.tensor(
                self.data_y[
                    index + self.seq_length : index + self.seq_length + self.pred_length
                ],
                dtype=torch.float,
            ),
        )


class CustomDartsDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset_name: str,
        batch_size: int = 12,
        **params,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.params = params

    def setup(self, stage=None):
        """
        Stage: fit, test, predict
        """
        series = get_darts_series(self.dataset_name)
        self.data_train = CustomDartsDataset(series, **self.params, mode="train")
        self.data_val = CustomDartsDataset(series, **self.params, mode="val")
        self.data_test = CustomDartsDataset(series, **self.params, mode="test")
        self.scaler = self.data_train.scaler
        self.dates_iso = self.data_test.dates_iso

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size, shuffle=False)


if __name__ == "__main__":
    params = {
        "seq_length": 336,
        "pred_length": 96,
        "task": "MB",
        "do_dataset_split_ratio": True,
        "dataset_split": (0.7, 0.1, 0.2),
        "scale": True,
    }

    # Test the dataset
    run_through = False
    if run_through:
        for dataset_name in dataset_list_darts.DARTS_DATA:
            dataset = get_darts_series(dataset_name)
            custom_dataset = CustomDartsDataset(dataset, **params)
    else:
        dataset = get_darts_series("etth1")
        custom_dataset = CustomDartsDataset(dataset, **params)
        dataloader = DataLoader(custom_dataset, batch_size=12, shuffle=True)
        first_batch = next(iter(dataloader))
        print(first_batch[0].shape)

    # Test the dataloader
    if run_through:
        for dataset_name in dataset_list_darts.DARTS_DATA:
            custom_data_module = CustomDartsDataModule(dataset_name, **params)
            custom_data_module.setup()
            train_loader = custom_data_module.train_dataloader()
            val_loader = custom_data_module.val_dataloader()
            test_loader = custom_data_module.test_dataloader()
    else:
        custom_data_module = CustomDartsDataModule("etth1", **params)
        custom_data_module.setup()
        train_loader = custom_data_module.train_dataloader()
        val_loader = custom_data_module.val_dataloader()
        test_loader = custom_data_module.test_dataloader()
        first_batch = next(iter(train_loader))
        print(first_batch[0].shape)
        print(first_batch[1].shape)
        first_batch = next(iter(val_loader))
        print(first_batch[0].shape)
        first_batch = next(iter(test_loader))
        print(first_batch[0].shape)
