import os
import warnings
import logging
import lightning as L

from lightning.pytorch.callbacks import (
    RichProgressBar,
    ModelCheckpoint,
    EarlyStopping,
)
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme

from sonnet.model_config import ModelConfig
from sonnet.data_config import DataConfig
from sonnet.utils.utils import set_seed

from omegaconf import DictConfig, OmegaConf
import hydra


warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def run_experiment(cfg: DictConfig):
    os.environ["HYDRA_FULL_ERROR"] = "1"
    set_seed(cfg.seed)

    ##### Callbacks #####
    # Initialize the progress bar
    progress_bar = RichProgressBar(
        theme=RichProgressBarTheme(
            description="green_yellow",
            progress_bar="green1",
            progress_bar_finished="green1",
            progress_bar_pulse="#6206E0",
            batch_progress="green_yellow",
            time="grey82",
            processing_speed="grey82",
            metrics="grey82",
            metrics_text_delimiter="\n",
            metrics_format=".3e",
        )
    )

    # Initialize the checkpoint callback
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_dir = hydra_cfg["runtime"]["output_dir"]
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=cfg.exp.ckpt_name,
        save_top_k=1,
        monitor=cfg.exp.monitor_metric,
    )

    # Initialize the early stopping callback
    early_stop_callback = EarlyStopping(
        monitor=cfg.exp.monitor_metric,
        patience=cfg.exp.patience,
        verbose=True,
        mode=cfg.exp.patience_mode,
        min_delta=cfg.exp.patience_delta,
    )

    if cfg.dataset.splits.train is not None:
        if sum(cfg.dataset.splits.values()) > 1:
            cfg.exp.do_dataset_split_ratio = False
            cfg.exp.dataset_split_numeric = tuple(cfg.dataset.splits.values())
        else:
            cfg.exp.do_dataset_split_ratio = True

    if cfg.model.name.startswith("seasonal"):
        cfg.exp.seq_length = max(cfg.dataset.seasonality, cfg.exp.seq_length)

    # Data params
    data_exp_params = {
        **OmegaConf.to_container(cfg.exp, resolve=True),
        "task": cfg.model.model_task,
        **(
            {"end_year": cfg.dataset.end_year, "total_year": cfg.dataset.total_year}
            if cfg.dataset.data_group == "ili_trends"
            else {}
        ),
    }

    # Load data
    data_register = DataConfig(
        name=cfg.dataset.data_group,
    )
    dataset = data_register.data_module_class(
        cfg.dataset.name,
        **data_exp_params,
    )
    dataset.setup()
    train_loader = dataset.train_dataloader()
    val_loader = dataset.val_dataloader()
    test_loader = dataset.test_dataloader()
    num_variables = (
        dataset.num_variables
        if cfg.dataset.num_variables is None
        else cfg.dataset.num_variables
    )

    # Load model
    model_register = ModelConfig(
        name=cfg.model.name,
    )

    # Model params: for constructing each model
    model_params = OmegaConf.create(
        {
            **OmegaConf.to_container(cfg.model.model_params, resolve=True),
            "seq_len": cfg.exp.seq_length,
            "pred_len": cfg.exp.pred_length,
            "enc_in": num_variables,
            "dec_in": num_variables,
            "seasonality": cfg.dataset.seasonality,
            "c_out": 1,
        }
    )

    # Exp params: for lightning exp module
    exp_params = {
        "lr": cfg.exp.learning_rate,
        "weight_decay": cfg.exp.weight_decay,
        "lr_scheduler": cfg.exp.lr_scheduler,
        "reverse_eval": cfg.exp.reverse_eval,
    }
    model = model_register.model_class(model_params, datamodule=dataset, **exp_params)

    # Train model
    if cfg.model.name.endswith("persistence"):
        trainer = L.Trainer(
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            devices=[0],
            logger=False,
        )
        trainer.test(model, test_loader)
    else:
        logger.info(f"Training model: {cfg.model.name}")
        trainer = L.Trainer(
            max_epochs=cfg.exp.epochs,
            callbacks=[progress_bar, checkpoint_callback, early_stop_callback],
            accelerator=cfg.exp.accelerator,
            devices=[cfg.exp.devices],
            logger=False,
        )
        trainer.fit(model, train_loader, val_loader)
        trainer.test(model, test_loader, ckpt_path=cfg.exp.ckpt_name)


if __name__ == "__main__":
    run_experiment()
