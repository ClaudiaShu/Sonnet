# Configuration Guide

This guide explains the current structure, purpose, and usage of configuration files within the project. It is intended to help users navigate and modify config files for various components.

## Folder Structure Overview

The repository organizes config files into subdirectories based on their role. A typical structure might look like this:

```
/path_to_code/Sonnet/
├── configs/
│   ├── base_config.yaml         # Base configurations shared by all modules
│   ├── model/
│   │   ├── model_a.yaml         # Configuration for Model A
│   │   └── model_b.yaml         # Configuration for Model B
│   ├── dataset/
│   │   ├── dataset_x       # Group of data X
│   │   │   ├── dataset_xx.yaml       # Dataset XX configurations
│   │   └── dataset_y       # Group of data Y
│   │   │   └── dataset_yy.yaml       # Dataset YY configurations
│   └── exp/
│       ├── model_a    # Folder that contains the best tuned HP for model A
│       ├── model_b    # Folder that contains the best tuned HP for model A
│       └── exp_1.yaml # Experiment configurations, including sequence length and learning rate etc.
```

## Hydra Configuration for Managing Project Configurations

This project configuration is based on Hydra. Hydra helps manage configurations dynamically and supports running experiments with different settings via command-line overrides. Below are some basics to guide you in using Hydra with the current folder structure.

### Running Experiments with Hydra

- Configure your main script (e.g., train.py or main.py) to use Hydra. Typically, you’d decorate your main function with `@hydra.main(config_path="configs", config_name="base_config")`.
- Use command-line arguments to override specific configuration values or switch configuration groups. For example:

    Run Model A with a specific dataset:
    ```
    python main.py model=model/model_a.yaml dataset=dataset/dataset_x/dataset_xx.yaml exp=exp/exp_1.yaml
    ```

- For experiments, you can specify the experiment folder as a configuration override if you have additional hyperparameters:
    ```
    python main.py exp=exp/model_a
    ```

### Organizing Configuration Groups

- **Base Configurations**: The `base_config.yaml` holds shared settings.
- **Model Configurations**: Located under `configs/model/`. Each model config can be selected by passing its relative path.
- **Dataset Configurations**: Under `configs/dataset/`. Organize dataset settings based on the grouped subdirectories.
- **Experiment Configurations**: Located in `configs/exp/`. They can contain experiment-specific hyperparameters.

### Tips for Using Hydra

- **Structured Overrides:** You can override any parameter. For example:
    ```
    python main.py model.learning_rate=0.001 dataset.batch_size=32
    ```
- **Multi-Run Experiments:** Hydra supports multi-run configuration for hyperparameter sweeps. Use the `-m` flag to execute multiple runs:
    ```
    python main.py -m model=“[model_a,model_b]” dataset=dataset/dataset_x/dataset_xx.yaml
    ```
- **Config Composition:** Hydra automatically composes configurations from multiple files. Use it to layer common defaults with module-specific settings.

With these guidelines, you can easily switch between models, datasets, and experiment configurations from the command line, making your workflow flexible and reproducible.
