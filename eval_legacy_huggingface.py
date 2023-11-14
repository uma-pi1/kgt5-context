import os
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import argparse
import pytorch_lightning as pl
from main import process_deprecated
from pytorch_lightning.loggers import WandbLogger
from custom_data import NbhoodDataModule
from kgt5_model import KGT5_Model
from omegaconf import DictConfig, OmegaConf, open_dict
from tqdm import tqdm


def move_to_device(batch, device):
    for key, value in batch.items():
        if type(value) is torch.Tensor:
            batch[key] = value.to(device)
    return batch


def run(checkpoint_path: str, dataset_name: str, v1: bool, is_legacy: bool, split: str, device: str, descriptions: bool) -> None:
    config = OmegaConf.load(os.path.join("conf", "config.yaml"))
    config.dataset.v1 = v1
    config.dataset.name = dataset_name
    config.dataset.is_legacy = is_legacy
    config.output_dir = checkpoint_path
    config = process_deprecated(config)
    config.eval.num_predictions = 500
    config.descriptions.use = descriptions
    print(OmegaConf.to_yaml(config))
    print("output written to", config.output_dir)

    dm = NbhoodDataModule(config=config)
    t5_model = AutoModelForSeq2SeqLM.from_pretrained(args["model"]).to(device)
    try:
        tokenizer = AutoTokenizer.from_pretrained(args["model"])
        dm.tokenizer = tokenizer
    except:
        print("no pretrained tokenizer stored with checkpoint, using default one.")

    model = KGT5_Model(config=config, data_module=dm).to(device)
    model.model = t5_model

    dm.setup(stage=split)
    if split == "valid":
        data_loader = dm.val_dataloader()
    else:
        data_loader = getattr(dm, f"{split}_dataloader")()
    rank_dicts = list()
    for batch in tqdm(data_loader):
        rank_dicts.append(model.evaluate(move_to_device(batch, device), mode=split))
    model.metric_aggregation(rank_dicts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test kgt5 model')
    parser.add_argument('-m', '--model', help='Path to checkpoint',
                        required=True)
    parser.add_argument('-s', '--split', help='Split to evaluate on',
                        default="test")
    parser.add_argument('-d', '--dataset', help='dataset name',
                        default="wikidata5m_v3")
    parser.add_argument('-dev', '--device', help='compute device',
                        default="cuda")
    parser.add_argument(
        '--v1',
        action='store_true',
        help="whether the model is the original KGT5 without context",
    )
    parser.add_argument(
        '--is_legacy',
        action='store_true',
        help="whether the old input format should be used",
    )
    parser.add_argument(
        '--descriptions',
        action='store_true',
        help="whether the old input format should be used",
    )
    args = vars(parser.parse_args())
    torch.set_float32_matmul_precision('medium')
    run(checkpoint_path=args["model"], dataset_name=args["dataset"], v1=args["v1"], is_legacy=args["is_legacy"], split=args["split"], device=args["device"], descriptions=args["descriptions"])



