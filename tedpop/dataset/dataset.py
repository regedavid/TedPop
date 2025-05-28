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
from sklearn.preprocessing import QuantileTransformer, MinMaxScaler

class TEDDataset(Dataset):
    def __init__(self, csv_file, text_column="transcript", target_column="viewCount", transform_type='log'):
        self.data = pd.read_csv(csv_file)
        self.text_column = text_column
        self.target_column = target_column
        self.transform_type = transform_type
        self.data = self.data.dropna(subset=[self.text_column, self.target_column])
        self.raw_targets = self.data[self.target_column].values.astype(np.float32)
        self._fit_transform_function()
        
        scaler = MinMaxScaler()
        self.data[self.target_column] = scaler.fit_transform(self.data[[self.target_column]])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = row[self.text_column]
        target = row[self.target_column]
        target = float(row[self.target_column])
        # target = torch.tensor(target, dtype=torch.float)
        target = self._transform_target(target)

        return {"text": text, "target": torch.tensor(target, dtype=torch.float32)}
    
    def _transform_target(self, target):
        if self.transform_type == "log":
            return np.log1p(target)
        elif self.transform_type == "log10":
            return np.log10(target + 1)
        elif self.transform_type == "zscore":
            return (target - self.mean) / self.std
        elif self.transform_type == "quantile":
            return self.quantile_transformer.transform([[target]])[0, 0]
        elif self.transform_type == "cbrt":
            return np.cbrt(target)
        else:
            return target  # no transform

    def inverse_transform(self, target_tensor):
    # Convert to float64 numpy array for safe math
        target = target_tensor.detach().cpu().numpy().astype(np.float64)

        if self.transform_type == "log":
            result = np.expm1(target)

        elif self.transform_type == "log10":
            result = (10 ** target) - 1

        elif self.transform_type == "zscore":
            result = (target * self.std.item()) + self.mean.item()

        elif self.transform_type == "quantile":
            result = self.quantile_transformer.inverse_transform(target.reshape(-1, 1)).flatten()

        elif self.transform_type == "cbrt":
            result = np.power(target, 3)
        else:
            result = target

        # Return as float32 Torch tensor
        return torch.tensor(result, dtype=torch.float32)
    
    # def inverse_transform(self, target):
    #     if self.transform_type == "log":
    #         return torch.expm1(target)
    #     elif self.transform_type == "log10":
    #         return torch.pow(10, target) - 1
    #     elif self.transform_type == "zscore":
    #         return (target * self.std) + self.mean
    #     elif self.transform_type == "quantile":
    #         target = target.detach().cpu().numpy()
    #         return self.quantile_transformer.inverse_transform(target.reshape(-1, 1)).flatten()
    #     elif self.transform_type == "cbrt":
    #         return torch.sign(target) * torch.pow(torch.abs(target), 3)
    #     else:
    #         return target
        
    def _fit_transform_function(self):
        if self.transform_type == "zscore":
            self.mean = self.raw_targets.mean()
            self.std = self.raw_targets.std()
        elif self.transform_type == "quantile":
            self.quantile_transformer = QuantileTransformer(n_quantiles=100, output_distribution="uniform", random_state=42)
            self.quantile_transformer.fit(self.raw_targets.reshape(-1, 1))


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
    def __init__(self, csv_file, audio_dir, text_column="transcript", target_column="viewCount", transform_type='log'):
        super().__init__(csv_file, text_column, target_column, transform_type)
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
    
class TEDDatasetQuantile(torch.utils.data.Dataset):
    def __init__(self, csv_file, text_column="transcript", target_column="viewCount", num_classes=10):
        self.data = pd.read_csv(csv_file)
        self.text_column = text_column
        self.target_column = target_column
        self.num_classes = num_classes
        self.data = self.data.dropna(subset=[self.text_column, self.target_column])
        scaler = MinMaxScaler()
        self.data[self.target_column] = scaler.fit_transform(self.data[[self.target_column]])

        self.data["popularity_class"] = pd.qcut(
            self.data[self.target_column],
            q=self.num_classes,
            labels=False
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = row[self.text_column]
        label = row["popularity_class"]
        return {
            "text": text,
            "target": torch.tensor(label, dtype=torch.long)
        }
        

class TEDMultimodalDatasetQuantile(TEDDataset):
    def __init__(self, csv_file, audio_dir, text_column="transcript", target_column="viewCount", num_classes=10):
        super().__init__(csv_file, text_column, target_column, num_classes)
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
    

def _dataset(trainer_type, model_type, **kwargs):
    if trainer_type == "regressor":
        if model_type == "text":
            valid_keys = {"csv_file", "text_column", "target_column", "transform_type"}
            return TEDDataset(**{k: v for k, v in kwargs.items() if k in valid_keys})

        elif model_type == "multimodal":
            valid_keys = {"csv_file", "text_column", "target_column", "transform_type", "audio_dir"}
            return TEDMultimodalDataset(**{k: v for k, v in kwargs.items() if k in valid_keys})

    elif trainer_type == "classifier":
        if model_type == "text":
            valid_keys = {"csv_file", "text_column", "target_column", "num_classes"}
            return TEDDatasetQuantile(**{k: v for k, v in kwargs.items() if k in valid_keys})

        elif model_type == "multimodal":
            valid_keys = {"csv_file", "text_column", "target_column", "num_classes", "audio_dir"}
            return TEDMultimodalDatasetQuantile(**{k: v for k, v in kwargs.items() if k in valid_keys})

    raise ValueError(f"Invalid combination: trainer_type={trainer_type}, model_type={model_type}")
