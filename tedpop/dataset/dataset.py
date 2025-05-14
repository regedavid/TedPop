import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pandas as pd
import librosa
import numpy as np
import os
import soundfile as sf
from torch.nn import functional as F

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
        elif self.transform_target == 'log10':


        return {"text": text, "target": target}
    
    def log_transform(self, target):
        target = torch.log1p(target)
        return target
    
    def inverse_log_transform(self, target):
        target = torch.expm1(target)
        return target
    
    def log10_transform(self, target):
        target = torch.log10(target + 1)
        return target
    
    def inverse_log10_transform(self, target):
        target = torch.pow(10, target) - 1
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

    batch_dict = {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "targets": targets
    }
    if "audio" in batch[0]:
        mfccs = [item["audio"] for item in batch]
        max_time = max(m.shape[1] for m in mfccs)

        # Pad each MFCC to (13, max_time)
        padded_mfccs = [
            F.pad(m, (0, max_time - m.shape[1]), mode="constant", value=0) for m in mfccs
        ]

        audio_batch = torch.stack(padded_mfccs)  # shape: (batch_size, 13, max_time)
        batch_dict["audio"] = audio_batch
    
    return batch_dict

class TEDMultimodalDataset(TEDDataset):
    def __init__(self, csv_file, audio_dir, text_column="transcript", target_column="viewCount", transform_target='log'):
        super().__init__(csv_file, text_column, target_column, transform_target)
        self.audio_dir = audio_dir

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        row = self.data.iloc[idx]
        
        talk_id = row['video_id']
        audio_path = os.path.join(self.audio_dir, f"{talk_id}.wav")
        audio_fea = self.compute_mfcc(audio_path)


        item["audio"] = torch.tensor(audio_fea, dtype=torch.float)
        return item
    
    def compute_mfcc(self, audio_path, sr=16000, n_mfcc=13):
        y, _ = librosa.load(audio_path, sr=sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

        return mfcc