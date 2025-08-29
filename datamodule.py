import os
import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader
import pytorch_lightning as pl
from prepare_assets import prepare_data


class BinaryIterableDataset(IterableDataset):
    """Yields single token sequences (x, y). The DataLoader will handle batching."""
    
    def __init__(self, file_path: str, block_size: int):
        super().__init__()
        self.file_path = file_path
        self.block_size = block_size

    def __iter__(self):
        data = np.memmap(self.file_path, dtype=np.uint16, mode='r')
        data_len = len(data)
        while True:
            i = torch.randint(data_len - self.block_size, (1,)).item()
            x = torch.from_numpy(data[i:i+self.block_size].astype(np.int64))
            y = torch.from_numpy(data[i+1:i+1+self.block_size].astype(np.int64))
            yield x, y


class GPTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, block_size: int, batch_size: int, num_workers: int = 0, runtime: str = "CPU"):
        super().__init__()
        self.save_hyperparameters()
        self.train_dataset = None
        self.val_dataset = None

    def prepare_data(self):
        prepare_data(self.hparams.data_dir)

    def setup(self, stage: str = None):
        train_path = os.path.join(self.hparams.data_dir, "train.bin")
        val_path = os.path.join(self.hparams.data_dir, "val.bin")
        
        self.train_dataset = BinaryIterableDataset(train_path, self.hparams.block_size)
        self.val_dataset = BinaryIterableDataset(val_path, self.hparams.block_size)

    def train_dataloader(self):
        use_pin_memory = (self.hparams.runtime == "GPU")
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=use_pin_memory,
            persistent_workers=(self.hparams.num_workers > 0),
            prefetch_factor=2 if self.hparams.num_workers > 0 else None,
            drop_last=True,
        )

    def val_dataloader(self):
        use_pin_memory = (self.hparams.runtime == "GPU")
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=use_pin_memory,
            persistent_workers=(self.hparams.num_workers > 0),
            prefetch_factor=2 if self.hparams.num_workers > 0 else None,
            drop_last=True,
        )