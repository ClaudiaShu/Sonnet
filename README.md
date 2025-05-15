
## Installing dependencies 

To install the requirements, it's best to make a new conda environment, you can do this by running:
```
conda env create -f environment.yaml
```
and then activate by running `conda activate bad-benchmarks`. Then run:
```
pip install -e .
```

To install the dev dependencies run 
```
pip install -r requirements.txt
```



## Running experiment

This repository is built using Hydra and PyTorch Lightning.

To run an experiment with a specific model (e.g., sonnet), use the following command:

```
python scripts/run_experiment.py model=sonnet
```

### Available Models

We support Sonnet and 10 baseline models
	•	sonnet
    •	deformtime
	•	crossformer
	•	moderntcn
	•	patchtst
	•	itransformer
    •	timexer
	•	samformer
	•	dlinear
    •	persistence
    •	seasonal_persistence
    

### Customizing model configurations

You can override specific configuration parameters directly from the command line. For example:

```
python scripts/run_experiment.py model=sonnet model.model_params.n_atoms=64 model.model_params.n_atoms=16 model.model_params.alpha=0.5
```

All configuration files can be found under `configs/model`


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

**WeatherBench**
Example on running weather prediction for Hong Kong in 2016:
```
python scripts/run_experiment.py dataset=weatherbench/hongkong/hongkong_2016 exp=weatherbench
```
We currently support experiments for the following 5 locations:
	•	hongkong
	•	london
	•	newyork
	•	singapore
	•	capetown

And 3 test seasons:
	•	2016
	•	2017
	•	2018

To switch locations or seasons, simply modify the dataset path. For example, for New York in 2017:
```
python scripts/run_experiment.py dataset=weatherbench/newyork/newyork_2017 exp=weatherbench
```
