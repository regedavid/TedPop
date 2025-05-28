import torch
import torch.nn as nn
import pytorch_lightning as pl
import argparse
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from pytorch_lightning import Trainer
from scipy.stats import spearmanr
from pytorch_lightning.loggers import TensorBoardLogger
from datetime import datetime

from tedpop.dataset.dataset import TEDDataset, TEDMultimodalDataset, TEDDatasetQuantile, collate_fn, _dataset
from tedpop.model.model import TextEncoder, AudioEncoder, TransformerTextEncoder

class TEDRegressor(pl.LightningModule):
    def __init__(self, text_encoder, audio_encoder=None, lr=2e-5, inverse_transform_fn=None):
        super().__init__()
        self.save_hyperparameters(ignore=["text_encoder", "audio_encoder", "inverse_transform_fn"])

        self.text_encoder = text_encoder
        self.audio_encoder = audio_encoder
        total_input_dim = self.text_encoder.output_dim
        if self.audio_encoder is not None:
            total_input_dim += self.audio_encoder.output_dim

        for param in self.text_encoder.bert.parameters():
            param.requires_grad = False
            
        self.regressor = nn.Sequential(
            nn.Linear(total_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

        self.loss_fn = nn.MSELoss()
        self.inverse_transform_fn = inverse_transform_fn

    def forward(self, input_ids, attention_mask, audio=None):
        text_fea = self.text_encoder(input_ids, src_mask=~attention_mask.bool())
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

        if self.inverse_transform_fn is not None:
            preds_exp = self.inverse_transform_fn(preds)
            targets_exp = self.inverse_transform_fn(targets)
            print("Target (original):", targets_exp[:5])
            print("Predicted:", preds_exp[:5])
            rmse = torch.sqrt(torch.mean((preds_exp - targets_exp) ** 2))
        else:
            rmse = torch.tensor(0.0, device=preds.device)

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
    
    
class TEDClassifier(pl.LightningModule):
    def __init__(self, text_encoder, audio_encoder=None, lr=2e-5, num_classes=10):
        super().__init__()
        self.save_hyperparameters(ignore=["text_encoder", "audio_encoder"])

        self.text_encoder = text_encoder
        self.audio_encoder = audio_encoder
        self.lr = lr
        self.num_classes = num_classes

        input_dim = self.text_encoder.output_dim
        if self.audio_encoder is not None:
            input_dim += self.audio_encoder.output_dim

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim//2),
            nn.ReLU(),
            nn.Linear(input_dim//2, num_classes)
        )

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, audio=None):
        text_fea = self.text_encoder(input_ids, src_mask=~attention_mask.bool())
        if self.audio_encoder is not None and audio is not None:
            audio_feat = self.audio_encoder(audio)
            features = torch.cat((text_fea, audio_feat), dim=1)
        else:
            features = text_fea
        logits = self.classifier(features)
        return logits

    def step(self, batch, stage):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        targets = batch["targets"]  # class labels
        audio = batch.get("audio")

        logits = self(input_ids, attention_mask, audio)
        loss = self.loss_fn(logits, targets)

        preds = torch.argmax(logits, dim=1)

        acc = (preds == targets).float().mean()

        try:
            rho, _ = spearmanr(preds.cpu().numpy(), targets.cpu().numpy())
        except Exception:
            rho = 0.0  # fallback if all targets or preds are constant

        self.log(f"{stage}_loss", loss, prog_bar=(stage == "val"))
        self.log(f"{stage}_acc", acc, prog_bar=True)
        self.log(f"{stage}_spearman", rho, prog_bar=True)

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
    parser.add_argument("--transform_type", type=str, default="log", choices=["log", "log10", "zscore", "quantile", "cbrt"], help="Transform type")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--precision", type=int, default=16, help="Mixed precision (16 or 32)")
    parser.add_argument("--device", type=str, default="gpu", choices=["cpu", "gpu"], help="Device to train on")
    parser.add_argument("--devices", type=int, default=1, help="Number of devices (GPUs/CPUs) to use")
    parser.add_argument("--trainer_type", type=str, default="classifier", choices=["classifier", "regressor"], help="Trainer type. Can be a classifier or regressor model")
    parser.add_argument("--run_name", type=str, default="baseline", help="Custom run folder name for TensorBoard")
    

    args = parser.parse_args()
    print(args.model_type, args.transform_type)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    if args.model_type == "multimodal":
        audio_encoder = AudioEncoder()
    else:
        audio_encoder = None
        
        train_dataset = _dataset(
            trainer_type=args.trainer_type,
            model_type=args.model_type,
            csv_file="tedpop/dataset/train_filtered.csv",
            audio_dir="audio_wav/",
            text_column="transcript",
            target_column="viewCount",
            transform_type=args.transform_type,
            num_classes=10
        )
        val_dataset = _dataset(
            trainer_type=args.trainer_type,
            model_type=args.model_type,
            csv_file="tedpop/dataset/val_filtered.csv",
            audio_dir="audio_wav/",
            text_column="transcript",
            target_column="viewCount",
            transform_type=args.transform_type,
            num_classes=10
        )
    collate = lambda batch: collate_fn(batch, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=args.minibatch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_dataset, batch_size=args.val_minibatch_size, shuffle=False, collate_fn=collate)
    if args.trainer_type == 'regressor':
        model = TEDRegressor(
            text_encoder=TextEncoder("bert-base-uncased"),
            audio_encoder=audio_encoder,
            lr=args.lr,
            inverse_transform_fn=train_dataset.inverse_transform
        )
    elif args.trainer_type == 'classifier':
        model = TEDClassifier(
            text_encoder=TransformerTextEncoder(vocab_size=tokenizer.vocab_size),
            audio_encoder=audio_encoder,
            lr=args.lr,
            num_classes=10
        )
    run_name = f"{args.run_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    logger = TensorBoardLogger(
        save_dir="lightning_logs",
        name=run_name,
        default_hp_metric=False
    )
    trainer = Trainer(
        max_epochs=args.epochs,
        accelerator=args.device,
        devices=args.devices,
        precision=args.precision,
        logger=logger
    )

    trainer.fit(model, train_loader, val_loader)
    