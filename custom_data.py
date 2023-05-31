import pytorch_lightning as pl
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from transformers import AutoTokenizer, T5TokenizerFast

from kg_dataset import KGCDataset, SplitDatasetWrapper


class NbhoodDataModule(pl.LightningDataModule):
    def __init__(self, config: DictConfig, split="train"):
        super().__init__()
        self.config = config
        self.split = split
        self.pad_token_id = 0
        self.dataset_name = self.config.dataset
        self.batch_size = self.config.train.batch_size
        #self.tokenizer = T5TokenizerFast.from_pretrained(self.config.model.name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.config.model.name,
            model_max_length=self.config.model.max_input_length
        )
        if self.config.model.tokenizer_type == 't5':
            self.vocab_size = 32128
        self.max_input_sequence_length = self.config.model.max_input_length
        self.max_output_sequence_length = self.config.model.max_output_length
        self.num_workers = self.config.train.num_workers
        
        print('loading nbhood dataset')
        self.dataset = KGCDataset.create(config=config, split=self.split)
        print('Dataset size: ', len(self.dataset))

    def prepare_data(self):
        # download
        return

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        self.train_dataset = SplitDatasetWrapper(self.dataset, split="train")
        if self.config.valid.tiny:
            self.valid_dataset = SplitDatasetWrapper(self.dataset, split="valid_tiny")
        else:
            self.valid_dataset = SplitDatasetWrapper(self.dataset, split="valid")
        self.test_dataset = SplitDatasetWrapper(self.dataset, split="test")

    def _tokenize(self, input):
        return self.tokenizer(
            input,
            padding=True,
            truncation=True,
            max_length=self.max_input_sequence_length,
            return_tensors="pt"
        )
        
    def _collate_fn(self, batch):
        inputs_tokenized = self._tokenize([b["input"] for b in batch])
        targets_tokenized = self._tokenize([b["target"] for b in batch])
        input_ids, attention_mask = inputs_tokenized.input_ids, inputs_tokenized.attention_mask
        labels = targets_tokenized.input_ids
        # for labels, set -100 for padding
        labels[labels == self.pad_token_id] = -100
        output = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        return output

    def _collate_fn_eval(self, batch):
        inputs_tokenized = self._tokenize([b["input"] for b in batch])
        targets = [b["target"] for b in batch]
        queries = [b["query"] for b in batch]
        is_tail_pred = [b["is_tail_pred"] for b in batch]
        input_ids, attention_mask = inputs_tokenized.input_ids, inputs_tokenized.attention_mask
        output = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "targets": targets,
            "queries": queries,
            "is_tail_pred": is_tail_pred,
        }
        return output

    def _common_dataloader(self, dataset, batch_size=32, shuffle=False, collate="_collate_fn"):
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=getattr(self, collate),
            num_workers=self.num_workers,
            # persistent_workers=True,
        )
        return data_loader

    def train_dataloader(self):
        return self._common_dataloader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True
        )

    def val_dataloader(self):
        return self._common_dataloader(
            self.valid_dataset, batch_size=self.config.eval.batch_size, collate="_collate_fn_eval"
        )

    def test_dataloader(self):
        return self._common_dataloader(
            self.test_dataset, batch_size=self.config.eval.batch_size, collate="_collate_fn_eval"
        )
