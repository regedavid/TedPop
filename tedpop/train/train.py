import torch
import torch.nn as nn
import pytorch_lightning as pl
import argparse
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from pytorch_lightning import Trainer

from tedpop.dataset.dataset import TEDDataset, TEDMultimodalDataset, collate_fn
from tedpop.model.model import TextEncoder, AudioEncoder

class TEDRegressor(pl.LightningModule):
    def __init__(self, text_encoder, audio_encoder=None, lr=2e-5):
        super().__init__()
        self.save_hyperparameters(ignore=["text_encoder", "audio_encoder"])

        self.text_encoder = text_encoder
        self.audio_encoder = audio_encoder
        total_input_dim = self.text_encoder.output_dim
        if self.audio_encoder is not None:
            total_input_dim += self.audio_encoder.output_dim

        self.regressor = nn.Sequential(
            nn.Linear(total_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.loss_fn = nn.MSELoss()

    def forward(self, input_ids, attention_mask, audio=None):
        text_fea = self.text_encoder(input_ids, attention_mask)
        if self.audio_encoder is not None and audio is not None:
            audio_fea = self.audio_encoder(audio)
            features = torch.cat((text_fea, audio_fea), dim=1)
        else:
            features = text_fea
        preds = self.regressor(features)
        return preds.squeeze(-1)

    def step(self, batch, stage):
        preds = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            audio=batch.get("audio")
        )
        targets = batch["targets"]
        loss = self.loss_fn(preds, targets)

        preds_exp = torch.expm1(preds)
        targets_exp = torch.expm1(targets)
        rmse = torch.sqrt(torch.mean((preds_exp - targets_exp) ** 2))

        self.log(f"{stage}_loss", loss, prog_bar=(stage == "val"))
        self.log(f"{stage}_rmse", rmse, prog_bar=(stage == "val"))
        return loss
    
    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        return optimizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TED popularity regressor")

    parser.add_argument("--minibatch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--val_minibatch_size", type=int, default=16, help="Validation batch size")
    parser.add_argument("--model_type", type=str, default="text", choices=["text", "multimodal"], help="Model type")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--precision", type=int, default=16, help="Mixed precision (16 or 32)")
    parser.add_argument("--device", type=str, default="gpu", choices=["cpu", "gpu"], help="Device to train on")
    parser.add_argument("--devices", type=int, default=1, help="Number of devices (GPUs/CPUs) to use")

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    if args.model_type == "multimodal":
        train_dataset = TEDMultimodalDataset(
            csv_file="tedpop/dataset/train_filtered.csv",
            audio_dir="audio_wav/",
            text_column="transcript",
            target_column="viewCount",
            transform_target='log'
        )
        val_dataset = TEDMultimodalDataset(
            csv_file="tedpop/dataset/val_filtered.csv",
            audio_dir="audio_wav/",
            text_column="transcript",
            target_column="viewCount",
            transform_target='log'
        )
        audio_encoder = AudioEncoder()
    else:
        train_dataset = TEDDataset(
            csv_file="tedpop/dataset/train_filtered.csv",
            text_column="transcript",
            target_column="viewCount",
            transform_target='log'
        )
        val_dataset = TEDDataset(
            csv_file="tedpop/dataset/val_filtered.csv",
            text_column="transcript",
            target_column="viewCount",
            transform_target='log'
        )
        audio_encoder = None
    collate = lambda batch: collate_fn(batch, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=args.minibatch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_dataset, batch_size=args.val_minibatch_size, shuffle=False, collate_fn=collate)

    model = TEDRegressor(
        text_encoder=TextEncoder("bert-base-uncased"),
        audio_encoder=audio_encoder,
        lr=args.lr
    )

    trainer = Trainer(
        max_epochs=args.epochs,
        accelerator=args.device,
        devices=args.devices,
        precision=args.precision,
    )

    trainer.fit(model, train_loader, val_loader)
    