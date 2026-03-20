<!-- [# Sonnet] -->

<p align="center">
    <img src="figure/sonnet_logo.png" alt="Sonnet logo" width="620" />
</p>

<p align="center">
    <a href="https://huggingface.co/datasets/ClaudiaShu/wea_mts">
        <img src="https://img.shields.io/badge/🤗%20Datasets-wea__mts-blue" alt="Hugging Face Dataset" />
    </a>
    &nbsp;
    <a href="https://arxiv.org/abs/2505.15312">
        <img src="https://img.shields.io/badge/arXiv-2505.15312-b31b1b.svg" alt="arXiv" />
    </a>
</p>

This repository contains the official implementation of methods and experiments presented in our paper titled "<a href="https://arxiv.org/abs/2505.15312"><strong>Sonnet: Spectral Operator Neural Network for Multivariable Time Series Forecasting</strong></a>". It also includes the weather data set formed by us for multivariate time series forecasting.

```
@article{shu2026sonnet,
    title        = {{Sonnet: Spectral Operator Neural Network for Multivariable Time Series Forecasting}},
    author       = {Shu, Yuxuan and Lampos, Vasileios},
    year         = 2026,
    journal      = {Proceedings of the AAAI Conference on Artificial Intelligence},
    volume       = 40,
    number       = 30,
    pages        = {25419--25427},
    doi          = {10.1609/aaai.v40i30.39736},
    url          = {https://ojs.aaai.org/index.php/AAAI/article/view/39736},
}
```

## News 🎉 
- **(03/2026)** Our paper is now officially published online on **AAAI**! 📄 Read it here: [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/39736).
- **(01/2026)** AAAI Oral presentation in Singapore!! The official recording is available here: [Zoom link](https://us06web.zoom.us/rec/play/fQJ1ATzPZWbcECvVQj5xyFi3T8681TzqugOupbPWZDeNVL0v4-0NzT2KTKVb88kmNAlzlGOpHlFt3HWM.3O-x8OfnxamelxNh?eagerLoadZvaPages=&accessLevel=meeting&canPlayFromShare=true&from=share_recording_detail&startTime=1769216854000&componentName=rec-play&originRequestUrl=https%3A%2F%2Fus06web.zoom.us%2Frec%2Fshare%2FrwfGzRo-3Ezfb5sO0X19SRslDiz7r0z1oWd7sGPYtxPbYLwrWoerOsm5PJfaYHfk.PVGtCAhPmwXOkhPA%3FstartTime%3D1769216854000).
- **(11/2025)** Released a notebook comparing Sonnet with Chronos2, a foundation model for multivariable forecasting.
- **(11/2025)** Our paper has been accepted for an **oral presentation** at **AAAI**! 🎉
- **(11/2025)** Released a quickstart notebook demonstrating training and inference.
- **(05/2025)** Paper released on arXiv.

## Important to note:

### Multivariable vs. Multivariate Forecasting
Multivariable forecasting needs to be differenciated from multivariate forecasting. Multivariable forecasting is to use multiple variables to predict 1 target variable (also known as forecasting the target with exogenous variables); whereas multivariate forecasting is to use multiple variables to predict multiple target variables [1]. 

### Why error evaluated on the target horizon
When forecasting $H$ steps ahead, we evaluate accuracy using $y_{t+H}$ rather than averaging errors over $y_1, \dots, y_{t+H-1}$. The reason behind this is that forecasting errors tend to increase as the forecasting horizon extends and the forecasting task becomes harder [2], and models may perform differently at short versus long horizons. Averaging across all outputs can favor models that are accurate early but poor at the target horizon. For example, consider forecasting 3 steps ahead with two models: Model A has MAEs {1,5,9}, and Model B {4,5,6}. Both have an average MAE of 5, but Model B is more accurate at the actual target horizon (step 3).

- [1] Hidalgo, B., & Goodman, M. (2013). Multivariate or multivariable regression?
- [2] Hyndman, R. J., & Koehler, A. B. (2006). Another look at measures of forecast accuracy. 


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

You can overwrite specific configuration parameters directly from the command line. For example:

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
For the weather dataset, we currently support experiments for the following 5 locations:
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
