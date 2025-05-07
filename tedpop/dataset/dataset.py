import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pandas as pd

class TEDDataset(Dataset):
    def __init__(self, csv_file, text_column="transcript", target_column="viewCount", transform_target='log'):
        self.data = pd.read_csv(csv_file)
        self.text_column = text_column
        self.target_column = target_column
        self.transform_target = transform_target
        self.data = self.data.dropna(subset=[self.text_column, self.target_column])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = row[self.text_column]
        target = row[self.target_column]
        target = torch.tensor(target, dtype=torch.float)
        if self.transform_target == 'log':
            target = self.log_transform(target)

        return {"text": text, "target": target}
    
    def log_transform(self, target):
        target = torch.log1p(target)
        return target
    
    def inverse_log_transform(self, target):
        target = torch.expm1(target)
        return target

def collate_fn(batch, tokenizer, max_length=512):
    texts = [item["text"] for item in batch]
    targets = torch.stack([item["target"] for item in batch])

    tokenized = tokenizer(
        texts,
        padding="longest",  # Or "max_length" if you want fixed
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "targets": targets
    }