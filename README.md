# Sonnet

This repository contains the official implementation of methods and experiments presented in our paper titled "<strong>Sonnet: Spectral Operator Neural Network for Multivariable Time Series Forecasting</strong>". It also includes the weather data set formed by us for multivariate time series forecasting.

## Abstract

Multivariable time series forecasting methods can integrate information from exogenous variables, leading to significant prediction accuracy gains. Transformer architecture has been widely applied in various time series forecasting models due to its ability to capture long-range sequential dependencies. However, a naïve application of transformers often struggles to effectively model complex relationships among variables over time. To mitigate against this, we propose a novel architecture, namely the **S**pectral **O**perator **N**eural **Net**work (Sonnet). Sonnet applies learnable wavelet transformations to the input and incorporates spectral analysis using the Koopman operator. Its predictive skill relies on the **M**ulti**v**ariable **C**oherence **A**ttention (MVCA), an operation that leverages spectral coherence to model variable dependencies. Our empirical analysis shows that Sonnet yields the best performance on 34 out of 47 forecasting tasks with an average mean absolute error (MAE) reduction of 1.1\% against the most competitive baseline (different per task). We further show that MVCA---when put in place of the naïve attention used in various deep learning models---can remedy its deficiencies, reducing MAE by 10.7\% on average in the most challenging forecasting tasks.

## Highlights

We propose <span class="small-caps">Sonnet</span>, a novel multivariable time series (MTS) forecasting model that captures time series dependencies in the spectral domain using a learnable wavelet transform. We further introduce **MVCA**, an attention mechanism designed to model interactions between variables by leveraging their spectral coherence

We form a weather data set for MTS forecasting tasks with T850 (850 hPa temperature) as the target variable. Climate indicators from the same location and surrounding regions serve as exogenous variables. We select 5 spatially diverse cities from the grid: London (UK), New York (US), Hong Kong (China), Cape Town (South Africa), and Singapore, to capture different climatic conditions. All data covers the period from 1970 to 2018.

## Setting Up the Environment 

We recommend using **conda** to manage dependencies. To set up the environment:

1.	Create the environment using the provided environment.yaml file:
```
conda env create -f environment.yaml
```

2.	Activate the environment:
```
conda activate sonnet
```

3.	Install the package:
```
pip install -e .
```

4. To install additional development dependencies, run:
```
pip install -r requirements.txt
```

## Data sets

### Weather data sets

The **weather (WEA)** dataset used in our experiments is located in the following directory:

```
    datasets/
    ├──weatherbench/
```


### Darts data sets

The **electricity consumption (ELEC)** and **energy price (ENER)** data sets are obtained from [Darts](https://github.com/unit8co/darts). You do not need to download these data sets to run the experiments -- they are accessible via the Darts API, which is already integrated into our codebase.

If you want to use it for your own experiment, you can load them using:

```
from darts.datasets import EnergyDataset, ElectricityConsumptionZurichDataset
```


### ETT data sets

The ETT datasets (**ETTh1** and **ETTh2**) are also available through the Darts API.

You can load them using:

```
from darts.datasets import ETTh1Dataset, ETTh2Dataset
```

### ILI data sets

Due to data sharing restrictions, we are currently unable to provide the full version of the ILI datasets used in our experiments.

For detailed information on data accessibility and the Google search query pool, please refer to the [DeformTime](https://github.com/ClaudiaShu/DeformTime) codebase.


## Running experiment

This repository is built using Hydra and PyTorch Lightning.

To run an experiment with a specific model (e.g., sonnet), use the following command:

```
python scripts/run_experiment.py model=sonnet
```

### Available Models

We support Sonnet and 10 baseline models
- sonnet
- deformtime
- crossformer
- moderntcn
- patchtst
- itransformer
- timexer
- samformer
- dlinear
- persistence
- seasonal_persistence
    

### Customising model configurations

You can override specific configuration parameters directly from the command line. For example:

```
python scripts/run_experiment.py model=sonnet model.model_params.d_model=64 model.model_params.n_atoms=16 model.model_params.alpha=0.5
```


### Choosing data sets

You can run experiments on a variety of datasets by specifying the dataset path and experiment tag via command-line arguments.

**ELEC:**
Running with Zurich electricity consumption data for 2021:
```
python scripts/run_experiment.py dataset=exp_data_config/electricity_consumption_zurich_2021 exp=electricity
```
To experiment with year 2020, simply replace 2021 with the 2020:
```
python scripts/run_experiment.py dataset=exp_data_config/electricity_consumption_zurich_2020 exp=electricity
```

**ENER:**
```
python scripts/run_experiment.py dataset=exp_data_config/energy exp=energy
```

**ETTh1:**
```
python scripts/run_experiment.py dataset=exp_data_config/etth1 exp=etth
```

**ETTh2:**
```
python scripts/run_experiment.py dataset=exp_data_config/etth2 exp=etth
```

**WEA:**
Example on running weather prediction for Hong Kong in 2016:
```
python scripts/run_experiment.py dataset=weatherbench/hongkong/hongkong_2016 exp=weatherbench
```
We currently support experiments for the following 5 locations:
- hongkong
- london
- newyork
- singapore
- capetown

And 3 test seasons:
- 2016
- 2017
- 2018

To change the locations or test seasons, simply modify the dataset. For example, for New York in 2017, run:
```
python scripts/run_experiment.py dataset=weatherbench/newyork/newyork_2017 exp=weatherbench
```


## Reproducing the results

We note that the default parameter settings may not reproduce the exact results reported in the paper, as those results were obtained through hyperparameter tuning.

To support reproducibility, we provide scripts to replicate the results presented in the paper. However, due to potential variations across hardware or environments, we recommend performing hyperparameter tuning on your specific device for optimal performance.


To generate the run scripts for the WEA tasks, run the following command:

```
bash scripts/generate_runs.sh
```

This will create runfiles for each forecasting season and location under the `scripts/runs` folder.


## Acknowledgements

- The original WeatherBench dataset is available for download at [WeatherBench](https://dataserv.ub.tum.de/index.php/s/m1524895).
- We acknowledge [Darts](https://github.com/unit8co/darts) for their valuable data for time series forecasting.
- We also acknowledge [Informer](https://github.com/zhouhaoyi/Informer2020) for their data for time series forecasting.