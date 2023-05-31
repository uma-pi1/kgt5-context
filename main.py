import torch
import hydra
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from custom_data import NbhoodDataModule
from kgt5_model import KGT5_Model
from omegaconf import DictConfig, OmegaConf, open_dict


@hydra.main(version_base=None, config_path="conf", config_name="config")
def run(config: DictConfig) -> None:
    print(OmegaConf.to_yaml(config))
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    with open_dict(config):
        config.output_dir = hydra_cfg["runtime"]["output_dir"]
    config = process_deprecated(config)
    print("output written to", config.output_dir)

    dm = NbhoodDataModule(config=config)

    if len(config.resume_from) != 0:
        model = KGT5_Model.load_from_checkpoint(
            config.resume_from, config=config, data_module=dm
        )
    else:
        model = KGT5_Model(config, data_module=dm)

    checkpoint_monitor = ModelCheckpoint(
        filename="{epoch}-{step}",
        monitor="epoch",
        mode="max",
        save_top_k=config.checkpoint.keep_top_k,
    )

    train_options = {
        'accelerator': config.train.accelerator,
        'devices': config.train.devices,
        'max_epochs': config.train.max_epochs,
        'default_root_dir': config.output_dir,
        'strategy': config.train.strategy,
        'precision': config.train.precision,
        'callbacks': [checkpoint_monitor],
        'check_val_every_n_epoch': config.valid.every,
    }
    if config.wandb.use:
        wandb_logger = WandbLogger(
            name=config.wandb.run_name,
            project=config.wandb.project_name,
            config=config,
            save_dir=config.output_dir
        )
        #wandb_logger.experiment.config.update(config)
        train_options["logger"] = wandb_logger

    trainer = pl.Trainer(**train_options, )#num_sanity_val_steps=0,)# val_check_interval=100)# limit_train_batches=1)

    if len(config.resume_from) != 0:
        trainer.fit(model, dm, ckpt_path=config.resume_from)  # , ckpt_path=ckpt_path)
    else:
        trainer.fit(model, dm)  # , ckpt_path=ckpt_path)


def process_deprecated(config):
    if hasattr(config, "use_neighborhood"):
        config.context.use = config.use_neighborhood
        del config.use_neighborhood
    if hasattr(config, "use_wandb"):
        config.wandb.use = config.use_wandb
        del config.use_wandb
    if not hasattr(config, "descriptions"):
        config.descriptions = {"use": False}
    return config


if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    run()

    
    
